# vLLM-Ascend source snapshot

- Upstream: https://github.com/vllm-project/vllm-ascend
- Release: `v0.23.0.post1`
- Commit: `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`
- License: Apache-2.0; the upstream `LICENSE` is retained in `vllm-ascend/`.

The directory is a source snapshot of that upstream commit, not AdapDomain
code. It is included as regular files so GitHub displays and clones the source
without submodule initialization. Upstream's `csrc/third_party/catlass` is a
separate Git submodule and is not present in this archive. Builds that require
it must obtain the dependency at the revision declared in the upstream
`vllm-ascend/.gitmodules`. This snapshot has not been built or validated on
an Ascend host in this repository.
