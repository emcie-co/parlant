# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from lagom import Container

from parlant.core.common import SchemaVersion


class DocumentStoreVersionError(Exception):
    def __init__(
        self,
        database_name: str,
        expected_version: SchemaVersion,
        actual_version: SchemaVersion,
    ) -> None:
        self.database_name = database_name
        self.expected_version = expected_version
        self.actual_version = actual_version

    def __str__(self) -> str:
        extra = ""
        if self.expected_version > self.actual_version:
            extra = "[try the `parlant-server migrate` command to migrate the schemas]"
        return f"`{self.database_name}`: document store expects version={self.expected_version}, but version={self.actual_version} was found in the database. {extra}"


def validate_schema_versions(container: Container) -> list[Exception]:
    errs: list[Exception] = []
    visited: set[type] = set()
    for candidate_type in container.defined_types:
        candidate_instance = container[candidate_type]
        concrete_type = type(candidate_instance)

        if concrete_type in visited:
            continue

        if hasattr(candidate_instance, "VERSION"):
            visited.add(concrete_type)
            version_code: SchemaVersion = getattr(candidate_instance, "VERSION")
            version_fs: SchemaVersion = getattr(candidate_instance, "database_version")
            if version_code != version_fs:
                db_name: str = getattr(candidate_instance, "database_name")
                errs.append(
                    DocumentStoreVersionError(
                        db_name,
                        version_code,
                        version_fs,
                    )
                )

    return errs


DatabaseVersions = dict[str, SchemaVersion]
