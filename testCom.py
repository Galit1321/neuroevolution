import filecmp

with open('me.pred', 'r') as file1:
    with open('sapir.pred', 'r') as file2:
        same = set(file1).intersection(file2)

same.discard('\n')

with open('some_output_file.txt', 'w') as file_out:
    for line in same:
        file_out.write(line)