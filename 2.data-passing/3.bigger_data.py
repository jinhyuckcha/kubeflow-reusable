"""
https://github.com/kubeflow/pipelines/blob/master/samples/tutorials/Data%20passing%20in%20python%20components.ipynb
"""
from kfp.components import func_to_container_op, InputPath, OutputPath

"""
2. Bigger data(files)
Bigger data should be read from files and written to files.
The paths for the input and output files are chosen by the system and are passed into the function (as strings).

Use the InputPath parameter annotation to tell the system that the function wants to consume the corresponding input data as a file. 
The system will download the data, write it to a local file and then pass the path of that file to the function.
Use the OutputPath parameter annotation to tell the system that the function wants to produce the corresponding output data as a file. 

After the function exits, the system will upload the data to the storage system so that it can be passed to downstream components.
You can specify the type of the consumed/produced data by specifying the type argument to InputPath and OutputPath. 
OutputPath('TFModel') means that the function states that the data it has written to a file has type 'TFModel'. 
InputPath('TFModel') means that the function states that it expect the data it reads from a file to have type 'TFModel'. 
"""


# Writing bigger data
@func_to_container_op
def repeat_line(line: str, output_text_path: OutputPath(str), count: int = 10):
    with open(output_text_path, 'w') as writer:
        for i in range(count):
            writer.write(line + '\n')


# Reading bigger data
@func_to_container_op
def print_text(text_path: InputPath()):
    with open(text_path, 'r') as reader:
        for line in reader:
            print(line, end='')


def pipeline():
    repeat_lines_task = repeat_line(line='Hello', count=5000)
    print_text(repeat_lines_task.outputs)


if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(pipeline,  __file__[:-3] + ".tar.gz")
