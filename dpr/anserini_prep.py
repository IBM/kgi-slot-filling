from util.line_corpus import jsonl_lines, write_open
import ujson as json
import re
from util.args_help import fill_from_args
import os


class Options:
    def __init__(self):
        self.input = ''
        self.output = ''
        self.file_count = 8


_WHITESPACE = re.compile(r'\s+')

opts = Options()
fill_from_args(opts)

# pid, title, text -> id, contents
outfiles = [write_open(os.path.join(opts.output, f'{j}.jsonl')) for j in range(opts.file_count)]
for line_ndx, line in enumerate(jsonl_lines(opts.input)):
    jobj = json.loads(line)
    f = outfiles[line_ndx % len(outfiles)]
    f.write(json.dumps({'id': jobj['pid'],
                        'contents': _WHITESPACE.sub(' ', jobj['title'])+'\n\n'+jobj['text']})+'\n')
for of in outfiles:
    of.close()
