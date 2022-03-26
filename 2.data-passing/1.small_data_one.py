"""
https://github.com/kubeflow/pipelines/blob/master/samples/tutorials/Data%20passing%20in%20python%20components.ipynb
"""
from kfp.components import func_to_container_op
"""
1. Small data
Small data is the data that you'll be comfortable passing as program's command-line argument. 
Small data size should not exceed few kilobytes.
Some examples of typical types of small data are: number, URL, small string (e.g. column name).
All small data outputs will be at some point serialized to strings and all small data input values will be at some point deserialized from strings (passed as command-line argumants). 
There are built-in serializers and deserializers for several common types (e.g. str, int, float, bool, list, dict). 
All other types of data need to be serialized manually before returning the data. 
"""


@func_to_container_op
def produce_one_small_output() -> str:
    return 'Hello world'


# Create component
@func_to_container_op
def print_small_text(text: str):
    print(text)


def pipeline():
    produce_task = produce_one_small_output()
    consume_task = print_small_text(produce_task.outputs['output'])


if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(pipeline,  __file__[:-3] + ".tar.gz")
