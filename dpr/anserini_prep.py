from util.line_corpus import jsonl_lines, write_open
import ujson as json
import re
from util.args_help import fill_from_args


class Options:
    def __init__(self):
        self.input = ''
        self.output = ''


_WHITESPACE = re.compile(r'\s+')

opts = Options()
fill_from_args(opts)

# pid, title, text -> id, contents
with write_open(opts.output) as f:
    for line in jsonl_lines(opts.input):
        jobj = json.loads(line)
        f.write(json.dumps({'id': jobj['pid'],
                            'contents': _WHITESPACE.sub(' ', jobj['title'])+'\n\n'+jobj['text']})+'\n')
