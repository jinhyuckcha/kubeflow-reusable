"""
https://github.com/kubeflow/pipelines/tree/master/samples/core/volume_ops
"""
from kfp import dsl


@dsl.pipeline(
    name="VolumeOp Basic",
    description="A Basic Example on VolumeOp Usage."
)
def pipeline():
    vop = dsl.VolumeOp(
        name="create-pvc",
        resource_name="my-pvc",
        modes=dsl.VOLUME_MODE_RWO,
        size="1Gi"
    )

    cop = dsl.ContainerOp(
        name="cop",
        image="library/bash:4.4.23",
        command=["sh", "-c"],
        arguments=["echo foo > /mnt/file1"],
        pvolumes={"/mnt": vop.volume}
    )


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline, __file__ + ".zip")
