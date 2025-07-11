#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
from base64 import b64decode
from typing import Dict, List, Set

import yaml
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from revng.model import Binary, DiffSet  # type: ignore[attr-defined]
from revng.pipeline_description import PipelineDescription  # type: ignore[attr-defined]
from revng.pipeline_description import YamlLoader  # type: ignore[attr-defined]

from .project import Project


class DaemonProject(Project):
    """
    This class is used to run revng analysis and artifact through the
    revng daemon via GraphQL.
    """

    def __init__(self, url: str):
        self.client = Client(
            transport=AIOHTTPTransport(url=url),
            fetch_schema_from_transport=True,
            execute_timeout=None,
        )
        super().__init__()
        self._set_model(self._get_model())

    def set_binary_path(self, binary_path: str):
        query = gql(
            """
            mutation upload($file: Upload!) {
                uploadFile(file: $file, container: "input")
            }
        """
        )

        with open(binary_path, "rb") as binary_file:
            result = self.client.execute(
                query, variable_values={"file": binary_file}, upload_files=True
            )

        if not result["uploadFile"]:
            raise RuntimeError("File upload failed")

    def _get_artifact_impl(self, artifact_name: str, targets: Set[str]) -> Dict[str, bytes]:
        query = gql(
            """
            query($step: String!, $paths: String!, $index: BigInt!) {
                produceArtifacts(step: $step, paths: $paths, index: $index) {
                    __typename
                    ... on Produced {
                        result
                    }
                }
            }
        """
        )
        variables = {
            "step": artifact_name,
            "paths": ",".join(targets),
            "index": self._get_content_commit_index(),
        }

        response = self.client.execute(query, variable_values=variables)
        assert response["produceArtifacts"]["__typename"] == "Produced"
        data: Dict[str, str] = json.loads(response["produceArtifacts"]["result"])
        container_mime = self._get_artifact_container(artifact_name).MIMEType
        return {
            k.rsplit(":", 1)[0]: self._decode_result_on_mime(v, container_mime)
            for k, v in data.items()
        }

    def _mapped_artifact_mime(self, artifact_name: str) -> str:
        container_mime = self._get_artifact_container(artifact_name).MIMEType
        if container_mime.endswith("+tar+gz"):
            return container_mime.removesuffix("+tar+gz")
        raise ValueError

    def _analyze(
        self, analysis_name: str, targets: Dict[str, List[str]] = {}, options: Dict[str, str] = {}
    ):
        step = self._get_step_name(analysis_name)

        variables = {
            "step": step,
            "analysis": analysis_name,
            "container": json.dumps(targets),
            "options": json.dumps(options),
            "index": self._get_content_commit_index(),
        }

        query = gql(
            """
            mutation($step: String!, $analysis: String!, $container: String!,
                $options: String!, $index: BigInt!)
            {
                runAnalysis(step: $step, analysis: $analysis, options: $options,
                    containerToTargets: $container, index: $index)
                {
                    __typename
                }
            }
        """
        )

        response = self.client.execute(query, variable_values=variables)
        assert response["runAnalysis"]["__typename"] == "Diff"
        self._set_model(self._get_model())

    def _analyses_list(self, analysis_name: str):
        query = gql(
            """
            mutation($analysis: String!, $index: BigInt!) {
                runAnalysesList(name: $analysis, index: $index) {
                    __typename
                }
            }
        """
        )
        variables = {"analysis": analysis_name, "index": self._get_content_commit_index()}

        response = self.client.execute(query, variable_values=variables)
        assert response["runAnalysesList"]["__typename"] == "Diff"
        self._set_model(self._get_model())

    def _commit(self):
        diff = DiffSet.make(self._last_saved_model, self.model)
        if len(diff.Changes) == 0:
            return

        diff_yaml = diff.serialize()

        query = gql(
            """
            mutation ($options: String!, $index: BigInt!) {
                runAnalysis(step: "initial", analysis: "apply-diff", index: $index,
                options: $options) {
                    __typename
                    ... on Diff {
                        diff
                    }
                }
            }
        """
        )
        variables = {
            "options": json.dumps(
                {"apply-diff-global-name": "model.yml", "apply-diff-diff-content": diff_yaml}
            ),
            "index": self._get_content_commit_index(),
        }

        response = self.client.execute(query, variable_values=variables)
        assert response["runAnalysis"]["__typename"] == "Diff"
        self._last_saved_model = self._get_model()

    def _get_pipeline_description(self) -> PipelineDescription:
        response = self.client.execute(gql("{ pipelineDescription }"))
        return yaml.load(response["pipelineDescription"], Loader=YamlLoader)

    def _decode_result_on_mime(self, result: str, container_mime: str) -> bytes:
        if container_mime.startswith("text/") or container_mime == "image/svg":
            return result.encode("utf-8")
        else:
            return b64decode(result)

    def _get_model(self) -> Binary:
        response = self.client.execute(gql("""{ getGlobal(name: "model.yml") }"""))
        return Binary.deserialize(response["getGlobal"])

    def _get_content_commit_index(self) -> int:
        # TODO: Keep track of the commit index with websockets
        response = self.client.execute(gql("{ contextCommitIndex }"))
        return int(response["contextCommitIndex"])
