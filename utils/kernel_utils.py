import os
import re
import shutil
import glob

from dataclasses import dataclass
from typing import List, Tuple, Any
from collections import Counter

import mlx.core as mx

def remove_gputrace_file():
    gputrace_files = glob.glob("custom_kernel_*.gputrace")
    
    if not gputrace_files:
        print("No .gputrace files found to remove.")
        return

    for gputrace_file in gputrace_files:
        try:
            if os.path.isdir(gputrace_file):
                shutil.rmtree(gputrace_file)
            else:
                os.remove(gputrace_file)
            print(f"Removed {gputrace_file}")
        except PermissionError:
            print(f"Permission denied: Unable to remove {gputrace_file}")
            print("Attempting to change permissions...")
            try:
                os.chmod(gputrace_file, 0o777)
                if os.path.isdir(gputrace_file):
                    shutil.rmtree(gputrace_file)
                else:
                    os.remove(gputrace_file)
                print(f"Successfully removed {gputrace_file} after changing permissions")
            except Exception as e:
                print(f"Failed to remove {gputrace_file} even after changing permissions: {e}")
        except Exception as e:
            print(f"Error removing {gputrace_file}: {e}")

    # Final check
    remaining_files = glob.glob("custom_kernel_*.gputrace")
    if not remaining_files:
        print("All .gputrace files have been successfully removed")
    else:
        print(f"Warning: Some .gputrace files still exist: {remaining_files}")
        print("Manual intervention may be required.")

@dataclass
class MetalKernel:
    name: str
    input_names: List[str]
    output_names: List[str]
    header: str = ""
    source: str = ""

    def __call__(self):
        return mx.fast.metal_kernel(
            name=self.name,
            input_names=self.input_names,
            output_names=self.output_names,
            header=self.header,
            source=self.source,
        )

@dataclass
class MetalProblem:
    name: str
    fn: Any
    inputs: List[mx.array]
    output_shapes: Tuple[int, int]
    grid: Tuple[int, int, int] = (1,1,1)
    threadgroup: Tuple[int, int, int] = (1,1,1)
    spec: Any = None

    def run_metal(self):

        assert mx.metal.is_available(), "Metal is not available"

        outputs = self.metalKernel()(
            inputs=self.inputs,
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes=[self.output_shapes],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
            verbose=os.getenv("VERBOSE") == '1',
            init_value=0,
        )

        return outputs[0]
    
    def score(self, results):
        total = 0
        full = Counter()
        for pos, (tt, a, c, out) in results[Coord(0, 0)].items():
            total += 1
            count = Counter()
            for out, tab in [(False, c2.refs[i]) for i in range(1, c.rounds()) for c2 in c.caches] + [(True, out)]:
                for inc in tab.incoming:
                    if out:
                        count["out_writes"] += 1
                    else:
                        count["shared_writes"] += 1
                    for ins in inc[1].inputs:
                        if ins.location[0].startswith("S"):
                            count["shared_reads"] += 1
                        else:
                            count["in_reads"] += 1
            for k in count:
                if count[k] > full[k]:
                    full[k] = count[k]
        print(f"""# {self.name}
 
   Score (Max Per Thread):
   | {'Global Reads':>13} | {'Global Writes':>13} | {'Shared Reads' :>13} | {'Shared Writes' :>13} |
   | {full['in_reads']:>13} | {full['out_writes']:>13} | {full['shared_reads']:>13} | {full['shared_writes']:>13} | 
        """) 

    def run_python(self):
        if self.threadgroup[0] == 1 and self.threadgroup[1] == 1:
            self.threadsperblock = Coord(self.grid[0], self.grid[1])
            self.blockspergrid = Coord(1, 1)
        else:
            self.threadsperblock = Coord(self.threadgroup[0], self.threadgroup[1])
            self.blockspergrid = Coord(self.grid[0] // self.threadgroup[0], self.grid[1] // self.threadgroup[1])

        self.metalKernel = self.fn(*self.inputs)
        metal_py = convert_source_to_py(self.metalKernel.header + self.metalKernel.source)

        inputs = {}
        for i in range(len(self.inputs)):
            curr = self.inputs[i]
            name = self.metalKernel.input_names[i]
            inputs[name + "_shape"] = curr.shape
            inputs[name + "_ndim"] = curr.ndim
            inputs[name + "_strides"] = mx.array([curr.shape[0], 1])

        globals().update(inputs)

        results = {}
        for i, block in self.blockspergrid.enumerate():
            results[block] = {}
            for tt, pos in self.threadsperblock.enumerate():
                tables = []
                args = ["a", "b", "c", "d"]
                for i, inp in enumerate(self.inputs):
                    globals()[args[i]] = Table(args[i], inp)
                    tables.append(globals()[args[i]])
                out = mx.zeros(self.output_shapes)
                out = Table("out", out)
                metal = Metal(block, self.threadsperblock, pos, pos)

                exec(metal_py)

                metal.finish()
                results[block][pos] = (tt, tables, metal, out)

        return results

    def show(self):
        results = self.run_python()
        self.score(results)

    def check(self):
        try:
            self.metalKernel = self.fn(*self.inputs)

            if os.getenv("MTL_CAPTURE_ENABLED") == '1':
                mx.eval(*self.inputs)
                
                # Only remove existing .gputrace files if they exist
                if glob.glob("custom_kernel_*.gputrace"):
                    remove_gputrace_file()
                
                traceName = f"custom_kernel_{self.metalKernel.name}.gputrace"
                mx.metal.start_capture(traceName)
                for _ in range(2): mx.eval(self.run_metal())
                mx.metal.stop_capture()

            x = self.run_metal()
            y = self.spec(*self.inputs)

            if mx.allclose(x, y): 
                print("Passed Tests!")
                return 

            print("Failed Tests.")
            print("Yours:", x)
            print("Spec :", y)

        except AssertionError as e:
            print(f"Error: {e}")

def convert_source_to_py(source):
    metal_source = preprocess_source(source)

    output_lines = []
    incr_stack = []
    indent_level = 0
    statement = False
    lines = metal_source.splitlines()
    for line in lines:
        line = line.strip()

        if line.count("}") > 0 and line.count("{") > 0:
            pass
        elif line.count("{") > 0:
            indent_level += 1
            if line == "{": continue
        elif line.count("}") > 0:
            indent_level -= 1
            if line == "}": 
                if incr_stack:
                    incr_indent_level, incr_line = incr_stack[-1]
                    if incr_indent_level == indent_level + 1:
                        output_lines.append(incr_line)
                        incr_stack.pop()
                continue

        line_no_braces = line.replace('{', '').replace('}', '')

        for keyword, pattern, replacement in [
            ('else if', r'\s*else\s+if\s*\((.*)\)', r'elif \1:'),
            ('if', r'\s*if\s*\((.*)\)', r'if \1:'),
            ('while', r'\s*while\s*\((.*)\)', r'while \1:'),
            ('else', r'\s*else', r'else:')
        ]:
            if re.match(pattern, line_no_braces):
                line = re.sub(pattern, replacement, line_no_braces)
                statement = True
                break

        m = re.match(r'for\s*\(\s*(.*?);\s*(.*?);\s*(.*?)\s*\)\s*\{?', line)
        if m:
            init = m.group(1).strip()
            cond = m.group(2).strip()
            incr = m.group(3).strip()

            init = re.sub(r'()\s*=\s*(.*)', r'\1 = int(\2)', init)

            output_lines.append('    ' * (indent_level-1) + init)
            output_lines.append('    ' * (indent_level-1) + "while " + cond + ":")

            if '++' in incr:
                incr = re.sub(r'(\+\+)(\w+)', r'\2 += 1', incr)
                incr = re.sub(r'(\w+)(\+\+)', r'\1 += 1', incr)
            elif '--' in incr:
                incr = re.sub(r'(\-\-)(\w+)', r'\2 -= 1', incr)
                incr = re.sub(r'(\w+)(\-\-)', r'\1 -= 1', incr)
            elif '/=' in incr:
                incr = re.sub(r'\s*(.*?)\s*/=\s*(.*)', r'\1 = int(\1 / \2)', incr)

            incr_line = '    ' * indent_level + incr
            incr_stack.append((indent_level, incr_line))
            continue

        if not statement:
            line = line.replace(';', '')
            line = '    ' * indent_level + line
        else:
            line = '    ' * (indent_level-1) + line
        output_lines.append(line)
        statement = False

    return '\n'.join(output_lines)

def preprocess_source(source):
    source = re.sub(r'//.*', '', source)
    source = re.sub(r'threadgroup float (\w+)\[(\w+)\]\[(\w+)\];', r'\1 = metal.threadgroupMemory.array((\2, \3))', source)
    source = re.sub(r'threadgroup float (\w+)\[(\w+)\];', r'\1 = metal.threadgroupMemory.array(\2)', source)
    source = re.sub(r'threadgroup_barrier\(mem_flags::mem_threadgroup\);', 'metal.syncthreads()', source)
    source = re.sub(r'metal::', '', source)
    source = re.sub(r'\b(uint|int|float|double|auto|constant)\b', '', source)
    source = re.sub(r'\s*(\]\[)', ', ', source)
    source = source.replace('&&', 'and')
    source = source.replace('||', 'or')

    replacements = [
        ('thread_position_in_grid', 'metal.thread_position_in_grid'),
        ('threadgroup_position_in_grid', 'metal.threadgroup_position_in_grid'),
        ('threads_per_threadgroup', 'metal.threads_per_threadgroup'),
        ('thread_position_in_threadgroup', 'metal.thread_position_in_threadgroup')
    ]
    for old, new in replacements:
        source = re.sub(old, new, source)
    
    return source 


@dataclass
class ScalarHistory:
    last_fn: str
    inputs: list

    def __radd__(self, b):
        return self + b

    def __add__(self, b):
        if isinstance(b, (float, int)):
            return self
        if isinstance(b, Scalar):
            return ScalarHistory(self.last_fn, self.inputs + [b])
        if isinstance(b, ScalarHistory):
            return ScalarHistory(self.last_fn, self.inputs + b.inputs)
        return NotImplemented
        
class Scalar:
    def __init__(self, location):
        self.location = location

    def __mul__(self, b):
        if isinstance(b, (float, int)):
            return ScalarHistory("id", [self])
        if isinstance(b, Scalar):
            return ScalarHistory("*", [self, b])
        return NotImplemented

    def __radd__(self, b):
        return self + b
        
    def __add__(self, b):
        if isinstance(b, (float, int)):
            return ScalarHistory("id", [self])
        if isinstance(b, Scalar):
            return ScalarHistory("+", [self, b])
        if isinstance(b, ScalarHistory):
            return ScalarHistory("+", [self] + b.inputs)
        return NotImplemented
    
class Table:
    def __init__(self, name, array):
        self.name = name
        self.incoming = []
        self.array = array

        self.size = array.shape
    
    def __getitem__(self, index):
        if isinstance(index, int):
            index = (index // self.size[1], index % self.size[1]) if len(self.size) == 2 else (index,)
        assert len(index) == len(self.size), "Wrong number of indices"
        if index[0] >= self.size[0]:
            assert False, "bad size"

        return Scalar((self.name,) + index)

    def __setitem__(self, index, val):
        if isinstance(index, int):
            index = (index // self.size[1], index % self.size[1]) if len(self.size) == 2 else (index,)
        assert len(index) == len(self.size), "Wrong number of indices"
        if index[0] >= self.size[0]:
            assert False, "bad size"
        if isinstance(val, Scalar):
            val = ScalarHistory("id", [val])
        if isinstance(val, (float, int)):
            return
        assert isinstance(val, ScalarHistory), "Assigning an unrecognized value"
        self.incoming.append((index, val))

@dataclass(frozen=True, eq=True)
class Coord:
    x: int
    y: int

    def enumerate(self):
        k = 0
        for i in range(self.y):
            for j in range(self.x):
                yield k, Coord(j, i)
                k += 1

    def tuple(self):
        return (self.x, self.y)

class RefList:
    def __init__(self):
        self.refs = []
        
    def __getitem__(self, index):
        return self.refs[-1][index]

    def __setitem__(self, index, val):
        self.refs[-1][index] = val


class ThreadgroupMemory:
    def __init__(self, metal):
        self.metal = metal

    def array(self, size):
        if isinstance(size, int):
            size = (size,)
        s = mx.zeros(size)
        cache = Table("S" + str(len(self.metal.caches)), s)
        # self.caches.append(cache)
        self.metal.caches.append(RefList())
        self.metal.caches[-1].refs = [cache]
        self.metal.saved.append([])
        return self.metal.caches[-1]


class Metal:
    threadgroup_position_in_grid: Coord
    threads_per_threadgroup: Coord
    thread_position_in_threadgroup: Coord
    thread_position_in_grid: Coord
    caches: list
    threadgroupMemory: ThreadgroupMemory

    def __init__(
        self, 
        threadgroup_position_in_grid,
        threads_per_threadgroup,
        thread_position_in_threadgroup,
        thread_position_in_grid
    ):
        self.threadgroup_position_in_grid = threadgroup_position_in_grid
        self.threads_per_threadgroup = threads_per_threadgroup
        self.thread_position_in_threadgroup = thread_position_in_threadgroup
        self.thread_position_in_grid = thread_position_in_grid
        self.caches = []
        self.threadgroupMemory = ThreadgroupMemory(self)
        self.saved = []

    def syncthreads(self):
        for i, c in enumerate(self.caches):
            old_cache = c.refs[-1]
            # self_links = cache.self_links()
            # cache.clean()
            temp = old_cache.incoming
            old_cache.incoming = self.saved[i]
            self.saved[i] = temp
            cache = Table(old_cache.name + "'", old_cache.array)

            c.refs.append(cache)

    def finish(self):
        for i, c in enumerate(self.caches):
            old_cache = c.refs[-1]
            old_cache.incoming = self.saved[i]

    def rounds(self):
        if len(self.caches) > 0:
            return len(self.caches[0].refs)
        else:
            return 0