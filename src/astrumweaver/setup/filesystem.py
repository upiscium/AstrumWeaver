"""Descriptor-relative filesystem operations for privileged setup.

A live filesystem trusts root-owned, non-writable ancestors. A staged root is
an explicit test/installation boundary owned by the invoking user. Runtime
*directory* operations may additionally traverse the service user's data, but
installer files never use that weaker policy. No existing object is adopted by
chmod/chown. All diagnostics deliberately omit target paths and file contents.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
from collections.abc import Iterator
from pathlib import Path


class UnsafeSetupPath(RuntimeError):
    """The requested filesystem operation crosses setup's trust boundary."""


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_UNSET = object()


class SetupFilesystem:
    def __init__(self, root: Path = Path("/")) -> None:
        if not root.is_absolute() or ".." in root.parts:
            raise UnsafeSetupPath("setup root must be absolute without parent traversal")
        self.root = root
        self.owner = 0 if root == Path("/") else os.geteuid()

    def _parts(self, path: Path) -> tuple[str, ...]:
        if not path.is_absolute() or ".." in path.parts:
            raise UnsafeSetupPath("setup path must be absolute without parent traversal")
        try:
            return path.relative_to(self.root).parts
        except ValueError:
            raise UnsafeSetupPath("setup path is outside the selected root") from None

    def _validate_directory(
        self, fd: int, owners: set[int], *, ancestor: bool = False,
    ) -> os.stat_result:
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid not in owners:
            raise UnsafeSetupPath("setup directory has unsafe type or ownership")
        # A root-owned sticky system temporary directory may be traversed;
        # every descendant is still separately opened and ownership-checked.
        sticky_system = ancestor and metadata.st_uid == 0 and bool(
            metadata.st_mode & stat.S_ISVTX
        )
        if metadata.st_mode & 0o022 and not sticky_system:
            raise UnsafeSetupPath("setup directory is writable by an untrusted group/user")
        return metadata

    def _open_root(self, *, create: bool) -> int:
        # Validate the staging anchor's ancestry too. An absolute open(root)
        # would follow links in components preceding its final component.
        fd = os.open("/", _DIRECTORY_FLAGS)
        try:
            parts = self.root.parts[1:]
            for index, part in enumerate(parts):
                final = index == len(parts) - 1
                self._validate_directory(fd, {0, self.owner}, ancestor=True)
                created = False
                if final and create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                        created = True
                    except FileExistsError:
                        pass
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    raise
                except OSError:
                    raise UnsafeSetupPath("setup root has linked or non-directory ancestry") from None
                os.close(fd)
                fd = child
                if created:
                    self._validate_directory(fd, {os.geteuid()})
                    os.fchmod(fd, 0o755)
            self._validate_directory(fd, {self.owner})
            return fd
        except BaseException:
            os.close(fd)
            raise

    @contextlib.contextmanager
    def directory(
        self,
        path: Path,
        *,
        create: bool = False,
        mode: int | None = None,
        uid: int | None = None,
        gid: int | None = None,
        data_owner: int | None = None,
    ) -> Iterator[int]:
        parts = self._parts(path)
        owners = {0, self.owner}
        if data_owner is not None:
            owners.add(data_owner)
        fd = self._open_root(create=create)
        try:
            for index, part in enumerate(parts):
                final = index == len(parts) - 1
                created = False
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                        created = True
                    except FileExistsError:
                        pass
                try:
                    child = os.open(part, _DIRECTORY_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    raise
                except OSError:
                    raise UnsafeSetupPath("setup refuses linked or non-directory ancestry") from None
                try:
                    if created:
                        metadata = self._validate_directory(child, {os.geteuid()})
                        if stat.S_IMODE(metadata.st_mode) != 0o700 or os.listdir(child):
                            raise UnsafeSetupPath("new setup directory was replaced")
                        if final and uid is not None and os.geteuid() == 0:
                            os.fchown(child, uid, -1 if gid is None else gid)
                        os.fchmod(child, mode if final and mode is not None else 0o755)
                    metadata = self._validate_directory(child, owners, ancestor=not final)
                    if final:
                        if uid is not None and metadata.st_uid != uid:
                            raise UnsafeSetupPath("setup directory owner is not canonical")
                        if gid is not None and metadata.st_gid != gid:
                            raise UnsafeSetupPath("setup directory group is not canonical")
                        if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
                            raise UnsafeSetupPath("setup directory mode is not canonical")
                except BaseException:
                    os.close(child)
                    raise
                os.close(fd)
                fd = child
            yield fd
        finally:
            os.close(fd)

    def _validate_file(self, metadata: os.stat_result) -> None:
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsafeSetupPath("setup target is not a regular non-symlink file")
        if metadata.st_uid != self.owner or metadata.st_mode & 0o022:
            raise UnsafeSetupPath("setup file has unsafe ownership or permissions")
        if metadata.st_nlink != 1:
            raise UnsafeSetupPath("setup refuses multiply linked authoritative files")

    def _file_state(self, parent: int, name: str) -> os.stat_result | None:
        try:
            metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return None
        self._validate_file(metadata)
        return metadata

    @staticmethod
    def _identity(metadata: os.stat_result | None) -> tuple[int, int, int, int] | None:
        if metadata is None:
            return None
        return metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_ctime_ns

    def read_text(self, path: Path) -> str | None:
        self._parts(path)
        try:
            with self.directory(path.parent) as parent:
                before = self._file_state(parent, path.name)
                if before is None:
                    return None
                try:
                    fd = os.open(path.name, _FILE_FLAGS, dir_fd=parent)
                except OSError:
                    raise UnsafeSetupPath("setup file changed while being opened") from None
                with os.fdopen(fd, "r", encoding="utf-8") as handle:
                    self._validate_file(os.fstat(handle.fileno()))
                    if self._identity(before) != self._identity(os.fstat(handle.fileno())):
                        raise UnsafeSetupPath("setup file changed while being opened")
                    content = handle.read()
                if self._identity(before) != self._identity(self._file_state(parent, path.name)):
                    raise UnsafeSetupPath("setup file changed while being read")
                return content
        except FileNotFoundError:
            return None

    def _check_directory_current(self, path: Path, fd: int) -> None:
        with self.directory(path) as current:
            left, right = os.fstat(fd), os.fstat(current)
            if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
                raise UnsafeSetupPath("setup directory changed during the operation")

    def write_text(
        self, path: Path, content: str, *, mode: int = 0o600,
        gid: int | None = None, preserve_metadata: bool = False,
    ) -> None:
        self._parts(path)
        with self.directory(path.parent, create=True) as parent:
            before = self._file_state(parent, path.name)
            temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
            fd = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600, dir_fd=parent,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    selected_gid = before.st_gid if preserve_metadata and before else gid
                    if os.geteuid() == 0:
                        os.fchown(handle.fileno(), self.owner, -1 if selected_gid is None else selected_gid)
                    selected_mode = stat.S_IMODE(before.st_mode) if preserve_metadata and before else mode
                    os.fchmod(handle.fileno(), selected_mode)
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                if self._identity(before) != self._identity(self._file_state(parent, path.name)):
                    raise UnsafeSetupPath("setup file changed before atomic replacement")
                self._check_directory_current(path.parent, parent)
                os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
                self._check_directory_current(path.parent, parent)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass

    def remove_file(self, path: Path, *, expected: str | object = _UNSET) -> None:
        self._parts(path)
        current = self.read_text(path)
        if current is None:
            return
        if expected is not _UNSET and current != expected:
            raise UnsafeSetupPath("setup refuses to remove a changed file")
        with self.directory(path.parent) as parent:
            self._file_state(parent, path.name)
            os.unlink(path.name, dir_fd=parent)
            os.fsync(parent)

    def remove_directory(self, path: Path, *, data_owner: int | None = None) -> bool:
        self._parts(path)
        try:
            with self.directory(path.parent, data_owner=data_owner) as parent:
                child = os.open(path.name, _DIRECTORY_FLAGS, dir_fd=parent)
                try:
                    self._validate_directory(child, {0, self.owner, *(() if data_owner is None else (data_owner,))})
                finally:
                    os.close(child)
                try:
                    os.rmdir(path.name, dir_fd=parent)
                except OSError as exc:
                    import errno
                    if exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
                        return False
                    raise
                return True
        except FileNotFoundError:
            return False

    def _link_state(self, parent: int, name: str) -> tuple[str | None, os.stat_result | None]:
        try:
            metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return None, None
        if not stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != self.owner:
            raise UnsafeSetupPath("NVIDIA driver bridge target is not a managed symlink")
        return os.readlink(name, dir_fd=parent), metadata

    def read_link(self, path: Path) -> str | None:
        self._parts(path)
        try:
            with self.directory(path.parent) as parent:
                return self._link_state(parent, path.name)[0]
        except FileNotFoundError:
            return None

    def replace_link(self, path: Path, target: str, *, expected: str | None) -> None:
        self._parts(path)
        with self.directory(path.parent) as parent:
            previous, before = self._link_state(parent, path.name)
            if previous != expected:
                raise UnsafeSetupPath("driver bridge changed before replacement")
            temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
            os.symlink(target, temporary, dir_fd=parent)
            try:
                if self._identity(before) != self._identity(self._link_state(parent, path.name)[1]):
                    raise UnsafeSetupPath("driver bridge changed before replacement")
                self._check_directory_current(path.parent, parent)
                os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
                self._check_directory_current(path.parent, parent)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass

    def remove_link(self, path: Path, *, expected: str) -> None:
        self._parts(path)
        with self.directory(path.parent) as parent:
            if self._link_state(parent, path.name)[0] != expected:
                raise UnsafeSetupPath("driver bridge changed before removal")
            os.unlink(path.name, dir_fd=parent)
            os.fsync(parent)
