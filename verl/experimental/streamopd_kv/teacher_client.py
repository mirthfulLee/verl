# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Teacher client whose load-balancer lease covers the entire KV session."""

from typing import Any

from verl.workers.rollout.llm_server import LLMServerClient


class StreamingTeacherClient(LLMServerClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher_stream_servers = {}

    async def stream_teacher_chunk(
        self,
        request_id: str,
        *,
        token_ids: list[int],
        sampling_params: dict[str, Any],
        terminal: bool,
    ) -> dict[str, Any]:
        """Append tokens to one sticky vLLM StreamingInput teacher session."""
        resident = self._teacher_stream_servers.get(request_id)
        if resident is None:
            resident = await self._acquire_server(request_id)
            self._teacher_stream_servers[request_id] = resident
        server_id, server = resident
        try:
            result = await server.stream_teacher_chunk.remote(
                request_id=request_id,
                token_ids=token_ids,
                sampling_params=sampling_params,
                terminal=terminal,
            )
        except BaseException:
            self._teacher_stream_servers.pop(request_id, None)
            self._release_server(server_id)
            raise
        if terminal:
            self._teacher_stream_servers.pop(request_id, None)
            self._release_server(server_id)
        return result
