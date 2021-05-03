"""
standard library tool-kit
this module contains commonly used functions to process and manipulate standard library objects
"""
import csv
import functools
import itertools
import json
import logging
import os
import pickle
import random
import re
import sys
import time
from collections import namedtuple, defaultdict, OrderedDict
from concurrent.futures import as_completed, ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import timedelta
from threading import Thread

import psutil
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from slibtk.core import *
import smtplib
import os
from email.message import EmailMessage
import sys
import traceback
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


# decorators ###########################################################################################################

def log_input():
    return log_input_and_output(input_flag=True, output_flag=False)


def log_output():
    return log_input_and_output(input_flag=False, output_flag=True)


def log_input_and_output(input_flag=True, output_flag=True, positional_input_index: int = 0,
                         kw_input_key: Optional[Hashable] = None):
    """logs the input (first positional argument) and output of decorated function, you can specify a specific kw
     arg to be logged as input by specifying its corresponding param key"""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            if input_flag:
                if kw_input_key:
                    # noinspection PyTypeChecker
                    input_arg = kwargs[kw_input_key]
                else:
                    input_arg = _get_positional_arg(args, kwargs, positional_input_index)
                logger.info(f'{func.__name__}: input={input_arg}'[:300])

            result = func(*args, **kwargs)
            if output_flag:
                logger.info(f'{func.__name__}: output={result}'[:300])
            return result

        return inner_wrapper

    return outer_wrapper


def _get_positional_arg(args, kwargs, index: int = 0) -> Any:
    """returns the first positional arg if there are any, if there are only kw args it returns the first kw arg"""
    try:
        input_arg = args[index]
    except KeyError:
        input_arg = list(kwargs.values())[index]
    return input_arg


def sleep_before(secs_before: float):
    """call the sleep function before the decorated function is called"""
    return sleep_before_and_after(secs_before=secs_before, secs_after=0)


def sleep_after(secs_after: float):
    """call the sleep function after the decorated function is called"""
    return sleep_before_and_after(secs_before=0, secs_after=secs_after)


def sleep_before_and_after(secs_before: float = 0, secs_after: float = 0):
    """call the sleep method before and after the decorated function is called, pass in the sleep duration in
    seconds. Default values are 0."""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            if secs_before:
                time.sleep(secs_before)
            result = func(*args, **kwargs)
            if secs_after:
                time.sleep(secs_after)
            return result

        return inner_wrapper

    return outer_wrapper


def timer(func):
    """decorator that logs the time taken for the decorated func to run"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start: float = time.time()
        result = func(*args, **kwargs)
        hr_time_elapsed: str = hr_secs(time.time() - start)
        logger.info(f'time taken {func.__name__}: {hr_time_elapsed}')
        return result

    return wrapper


def runtimes(arg_values: Sequence):
    """decorator that records the runtime (seconds) for several values of a single
    argument that is passed to the decorated func, returning the argument: second
    pairs in a dictionary"""

    def outer_wrapper(func: Callable):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            logger.info(f'monitoring runtimes for func={func.__name__}, values={arg_values}')
            times = {}
            for value in arg_values:
                start = time.time()
                func(value, *args, **kwargs)
                seconds = time.time() - start
                times[value] = seconds
                logger.info(f'param={value} seconds={seconds}')

            return times

        return inner_wrapper

    return outer_wrapper


def average_timer(n_runs: int):
    """decorator that logs the average time taken for `n_runs` of the decorated"""

    def outer_wrapper(func: Callable):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            times = []
            for _ in range(n_runs):
                start = time.time()
                func(*args, **kwargs)
                times.append(time.time() - start)

            mil = 1_000_000
            print(
                f'n_runs: {n_runs}, average time taken: {statistics.mean(times) * mil:.3f}us, min: {min(times) * mil:.3f}us')
            return func(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


class RunTimes:
    """
    stores run-times of methods in the programme and can display totals
    """
    times = defaultdict(list)

    @classmethod
    def add_time(cls, method_nm, runtime: float) -> None:
        cls.times[method_nm].append(runtime)

    @classmethod
    def _max_method_len(cls):
        return max([len(key) for key in cls.times.keys()])

    @classmethod
    def show_total_times(cls) -> None:
        """print print the total cumulative runtime of each decorated method"""
        for method, times in cls.times.items():
            print(f'{method:}:'.ljust(cls._max_method_len() + 1), sum(times))

    @classmethod
    def get_method_runtime(cls, method_nm: str = None):
        """stores the runtime of decorated callable in RunTimes"""

        def outer_wrapper(func):
            @functools.wraps(func)
            def inner_wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                name = method_nm if method_nm else func.__name__
                cls.add_time(name, time.time() - start)
                return result

            return inner_wrapper

        return outer_wrapper



def with_cache(path: PathOrStr, use_cache: bool = True, fname: Optional[str] = None) -> Callable:
    """
    decorator that will use a pickled version of a functions return if it exists when the function called else it will
    execute the function as normal
    Parameters
    ----------
    path: cache file location
    use_cache: flag that allows you to overwrite the default behavior of reading from the cache

    Returns
    -------
    outer_wrapper: the decorated function callable
    """
    path = Path(path)

    def _outer_wrapper(func):
        @functools.wraps(func)
        def _inner_wrapper(*args, **kwargs):
            if (path / f'{func.__name__}.pickle').exists() and use_cache:
                cached = read_pickle((path / f'{func.__name__}.pickle'))
                logger.info(f'loaded return from cache: {func.__name__}')
                return cached

            result = func(*args, **kwargs)
            write_pickle(result, (path / f'{func.__name__}.pickle'))
            if not (path / f'{func.__name__}.pickle').exists():
                logger.info(f'no cache exists')
            logger.info(f'saved return to cache: {func.__name__}')
            return result

        return _inner_wrapper

    return _outer_wrapper


# string tools #########################################################################################################

def re_no_decimal_places(s: str):
    return re.sub(r'(\d*)\.(\d*)', r'\1', s)


def re_n_decimal_places(s: str, n: int = 0) -> str:
    matches = re.findall(r'(\d+)(\.\d*)', s)
    for match in matches:
        s = s.replace(''.join(match), match[0] + match[1][:n + 1])
    return s if n > 0 else re_no_decimal_places(s)


def camel2snake(s: str) -> str:
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def snake2camel(s: str) -> str:
    snake_re = re.compile('_?[a-z]+_?')
    return ''.join([word.strip('_').capitalize() for word in snake_re.findall(s)])


def to_slug(s: str):
    """covert string to slug format
    before: Hello  world!!
    after: hello_world"""
    s = str(s)
    s = re.sub('[^\w\s]+', '', s)
    return '_'.join(s.lower().split())


def prefix_if_first_is_digit(s) -> str:
    """if the first item of a string is a digit prefix it with an underscore"""
    if s[0].isdigit():
        return '_' + s
    return s


def replace(text: str, items: Union[Dict, Tuple]) -> str:
    """Execute a sequence of find and replace pairs on a string."""
    if isinstance(items, dict): items = items.items()
    for k, v in items:
        text = text.replace(k, v)
    return text


def trim(string: str) -> str:
    """remove all escape characters and double white spaces"""
    return ' '.join(string.replace(u'\xa0', u' ').split())


def hr_bytes(n_bytes: int, binary=False, decimal_places=1):
    """return bytes in a human readable format"""
    if binary:
        factor, units = 1024, ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
    else:
        factor, units = 1000, ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    for unit in units:
        if n_bytes < factor:
            break
        n_bytes /= factor
    return f"{n_bytes:.{decimal_places}f}{unit}"


def hr_numbers(num: float, decimal_places: int = 1) -> str:
    """return number in human readable format"""
    scale = 1000
    units = ['', 'K', 'M', 'B', 'T']
    for unit in units:
        if abs(num) < scale:
            break
        num /= scale
    return f'{num:,.{decimal_places}f}{unit}'


def grep_(pattern: str, text: str, window: int = 30) -> List[str]:
    """text pattern matches of length plus and minus window"""
    matches = []
    for match in re.finditer(pattern, string=text, flags=re.IGNORECASE):
        matches.append(text[match.start() - window:match.start() + window])
    return matches


# iterable tools #######################################################################################################

def is_listy(x: Any) -> bool: return isinstance(x, (tuple, list))


def listify(x: Any) -> List:
    """Make `x` a list"""
    if isinstance(x, str): x = [x]
    if not isinstance(x, Iterable): x = [x]
    return list(x)


assert isinstance(listify(['hello', 'world']), list)
assert isinstance(listify('hello world'), list)
assert isinstance(listify(range(5)), list)


def uniqueify(items: Iterable) -> List:
    """remove duplicates from iterable and preserve order"""
    return list(OrderedDict.fromkeys(items).keys())


def recurse_sum(x):
    """recursively sum floats and ints in any combination of nested lists and dicts"""
    numbers = []

    def cache_numbers(x: Any) -> None:
        if isinstance(x, (float, int)): numbers.append(x)
        return x

    recurse(cache_numbers, x)
    return sum(numbers)


def recurse(func: Callable, x: Any, *args, **kwargs) -> Any:
    """recursively apply func to any combination of nested lists and dicts"""
    if isinstance(x, (list, tuple)): return [recurse(func, o, *args, **kwargs) for o in x]
    if isinstance(x, dict): return {k: recurse(func, v, *args, **kwargs) for k, v in x.items()}
    return func(x, *args, **kwargs)


def wildcard_filter(items: Iterable, query: str) -> List[str]:
    """filter a list by query, can accept a wildcard search"""
    return [x for x in items if re.search(query, str(x))]


def map_with_special_case(items: Iterable, func_first: Callable, func_everything_else: Callable) -> Iterable:
    """map a function to an iterable with a special case function for the first item in the iterable"""
    results = []
    for i, item in enumerate(items):
        if i > 0:
            results.append(func_everything_else(item))
        else:
            results.append(func_first(item))
    return results


def flattener(lst: List[List]) -> List:
    """flatten list"""
    return list(itertools.chain.from_iterable(lst))


def transpose_grid(matrix: List[List]) -> List[Tuple[Any]]:
    """transpose a list matrix and return it"""
    return list(zip(*matrix))


def make_every_combination(items) -> List[Tuple]:
    """return a list of every combination of the original list of items. For every subset lenght."""
    combinations = []
    for i in range(len(items) + 1):
        combinations.extend(
            list(itertools.combinations(items, i))
        )
    return combinations


# iterator / generator tools ###########################################################################################

def chunkify(seq: Sequence, size: int) -> Iterator[List]:
    """yield successive size-sized chunks from seq"""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def chunks_index(seq: List, size: int) -> Generator[Dict[int, List], None, None]:
    """yield successive size-sized chunks from seq in a dicts where the key is the ith batch"""
    ChunkIndex = namedtuple('ChunkIndex', ['chunk', 'index'])
    for i in range(0, len(seq), size):
        yield ChunkIndex(chunk=seq[i:i + size], index=i)


# dictionary tools #####################################################################################################

def dict_sample(d: Dict, n_samples: int = 5) -> Dict:
    """randomly select a subset of a dictionary"""
    keys = list(d)
    keys = [random.choice(keys) for _ in range(n_samples)]
    return {key: d[key] for key in keys}


# datetime tools #######################################################################################################

def current_date_minute() -> str:
    """return current date minute as a string e.g. '2021-02-02_0905'"""
    return str(datetime.now().replace(microsecond=0)).replace(':', '').replace(' ', '_')[:-2]


def make_tsub(n: int) -> Tuple[datetime.date, datetime.date]:
    """return start and end date of the the 365 day period prior to today with an n year lag"""
    assert n > 0, 'n must be positive'
    year_lag = timedelta(days=(366 * (n - 1)))
    end = datetime.now().date() - timedelta(days=1)
    start = end - timedelta(days=365)
    start, end = start - year_lag, end - year_lag
    assert (end - start).days == 365
    return start, end


def past_date(start: Optional[DateLike] = None, years: int = 0, months: int = 0, weeks: int = 0, days: int = 0,
              as_str: bool = False, **kwargs) -> DateLike:
    """get a date x months/years ago from today or start if specified"""
    if not start:
        start = datetime.now()
    if isinstance(start, str): start = datetime.fromisoformat(start)
    date = start - relativedelta(years=years, months=months, weeks=weeks, days=days, **kwargs)
    if as_str:
        return date.date().isoformat()
    return date


def hr_secs(secs: float) -> str:
    """format seconds human readable format hours:mins:seconds"""
    secs_per_hour: int = 3600
    secs_per_min: int = 60
    hours, remainder = divmod(secs, secs_per_hour)
    mins, seconds = divmod(remainder, secs_per_min)

    return f'{int(hours):02}:{int(mins):02}:{seconds:05.2f}'


def hr_secs_elapsed(start: float) -> str:
    """format seconds from elapsed since start in human readable format hours:mins:seconds"""
    return hr_secs(time.time() - start)


# io in outs ##############################################################################################################


class FileTXT:
    """abstract class representing a text file"""

    def __init__(self, path: Union[Path, str]):
        self.path = Path(path) if isinstance(path, str) else path

    def exists(self) -> bool:
        return self.path.exists()

    def show(self, num: int = 10) -> None:
        with self.path.open() as f:
            for i in range(num):
                print(next(f))

    def clear(self) -> None:
        with self.path.open('w'):
            pass


class ListTXT(FileTXT):
    """object to represent a text file containing a list"""

    def __init__(self, path: Union[Path, str]):
        super().__init__(path)

    def write(self, lines: List) -> None:
        """save a list as a test file where each element corresponds to a line in the text file"""
        with self.path.open('w') as f:
            for line in tqdm(lines):
                f.write(f'{str(line).strip()}\n')
        logger.info(f'saved txt file: {self.path}')

    def read(self) -> List:
        """parse text file into list"""
        with self.path.open() as f:
            lines = f.read().splitlines()
        return lines


class LogTXT(FileTXT):
    """simple log file object"""

    def __init__(self, path: Union[Path, str]):
        super().__init__(path)

    def write(self, line) -> None:
        with self.path.open('a') as f:
            f.write(f'{str(line).strip()}\n')

    def read(self) -> List:
        with self.path.open() as f:
            lines = f.read().splitlines()
        return lines

    def clear(self) -> None:
        with self.path.open('w'):
            pass


class FileCSV:
    """simple csv file object"""

    def __init__(self, path):
        self.path = path

    def write(self, seq):
        with self.path.open('w') as f:
            writer = csv.writer(f)
            for line in tqdm(seq):
                writer.writerow(line)

    def read(self) -> Iterator:
        with self.path.open() as f:
            reader = csv.reader(f, delimiter=',')
            rows = (row for row in reader)
        return rows


class IterLines:
    """lazy iterator for every file matched in a glob search"""

    def __init__(self, path: Path, pattern: str):
        self.path = path
        self.pattern = pattern
        self.files = path.rglob(pattern)

    def __iter__(self):
        for file in self.files:
            for line in file.open():
                yield line.split()


def date_versioned_dir(dst: Path) -> Path:
    """Make directory with name of current date in destination directory and return it"""
    versioned_dir = dst / str(datetime.now().date())
    versioned_dir.mkdir(exist_ok=True, parents=True)
    return versioned_dir


def next_fname(path: PathOrStr) -> Path:
    """return next incremental file that does not exist (path.root)_{next_num}.(path.suffix)"""
    path = Path(path)
    parent, stem, suffix = path.parent, path.stem, path.suffix
    i = 0
    while (parent / f'{stem}_{i:02}{suffix}').exists():
        i += 1
    return parent / f'{stem}_{i:02}{suffix}'


def extend_path_name(path: OptPathOrStr, ext: str) -> Optional[Path]:
    """concat string to the end of the path name but before the suffix"""
    return path.parent / (f'{path.stem}_{ext}{path.suffix}') if path else None


@log_input()
def write_pickle(obj, path: PathOrStr) -> None:
    """write object to a pickle on your computer"""
    path = Path(path)
    with path.open('wb') as f:
        pickle.dump(obj, f)


@log_output()
def read_pickle(path: PathOrStr):
    """return stored object from a pickle file"""
    path = Path(path)
    logger.info(f'reading pickle: {path}')
    with path.open('rb') as f:
        obj = pickle.load(f)
    return obj


def write_json(obj, path: PathOrStr) -> None:
    """write object as json"""
    path = Path(path)
    with path.open('w') as f:
        json.dump(obj, f)


def read_json(path: PathOrStr) -> Union[List, Dict]:
    """return stored object from a json file"""
    path = Path(path)
    with path.open() as f:
        obj = json.load(f)
    return obj


def read_jsonlines(filename: str) -> List[str]:
    """read a json lines file as a list of dictionaries"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.dumps(line))
    return data


def write_iterables(seq: Sequence[Sequence], path: Path):
    with path.open('w') as f:
        wr = csv.writer(f, delimiter=" ")
        wr.writerows(seq)


def input_polar(msg, default='y'):
    """get a polar input from user"""
    prompt = f'{msg}: [{default}]: '
    while True:
        user_input: str = input(prompt).lower()
        if user_input == '':
            return default
        elif user_input not in ['y', 'n']:
            print('Error: choose y or n')
            continue
        else:
            break
    return user_input


def input_text(msg, default):
    """get a text input from user. Prompt example=msg: [default]: """
    prompt = f'{msg}: [{default}]: '
    while True:
        user_input: str = input(prompt).strip()
        if user_input == '':
            return default
        else:
            break
    return user_input


def input_num_index(msg: str, choices: Dict[str, str], default='0') -> str:
    """get a numerical index from user"""
    num_index = list(choices.keys())
    num_index_comma_sep = ", ".join(num_index)
    choice_str = '\n'.join([f'{index} - {option}' for index, option in choices.items()]) + '\n'
    instruction = f'Choose from {num_index_comma_sep} [{default}]:'
    prompt = f'{msg}:\n' + choice_str + instruction
    while True:
        user_index = input(prompt)
        if user_index == '':
            return default
        elif user_index not in num_index:
            print(f'Error: choose from {num_index_comma_sep}')
            continue
        else:
            break
    return user_index


# multi-processing/threading tools #####################################################################################


@timer
def multiproc_iter(iterable: Sequence, func: Callable, n_chunks: int = 16) -> List:
    """
    process a sequence as batches on multiple cpus ie in parallel. Note is cpu utilisation drop and the
    function is still handing there is probably an error in func.
    Args:
        iterable: item to be processed
        func: callable applied to each chunk
        n_chunks: number of chunk iterable will be split into

    Returns:
        iterable mapped by func
    """
    logger.info(f'starting multi-processing with: {func.__name__}')
    size = len(iterable) // n_chunks
    chunks = chunkify(iterable, size)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as e:
        result = e.map(func, list(chunks))
    return [item for chunk in result for item in chunk]


def multi_proc_progress(iterable: Sequence, func: Callable, n_chunks: int = 16) -> List:
    logger.info(f'starting multi-processing with: {func.__name__}')
    size = len(iterable) // n_chunks
    chunks = chunkify(iterable, size)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(func, chunk) for chunk in chunks]
        results = []
        for f in tqdm(as_completed(futures), total=n_chunks):
            results.append(f.result())
    return [item for chunk in tqdm(results) for item in chunk]


stop_threads: bool = False


def terminal_timer(refresh_period: float = 1) -> None:
    """periodically log time elapsed to the terminal, this function is intended to be run as a thread which can
    be ended safely by updating a global flag."""
    start = time.time()
    global stop_threads
    while not stop_threads:
        time.sleep(refresh_period)
        secs_elapsed = int(time.time() - start)
        logger.info(f'time elapsed: {hr_secs(secs_elapsed)}')
    logger.info('timer thread ended.')
    stop_threads = False


def run_with_terminal_timer(refresh_period: float = 1):
    """
    decorator that periodically prints the time elapsed during the decorated functions runtime. It does this by starting
    a timer in a separate thread, the thread is ended safely then the decorated function finishes.
    Args:
        refresh_period: how often timer prints to terminal in seconds

    """

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            thread = Thread(target=terminal_timer, args=(refresh_period,))
            thread.start()

            result = func(*args, **kwargs)
            global stop_threads
            stop_threads = True
            return result

        return inner_wrapper

    return outer_wrapper


def server_request(query: str):
    print(f'starting: {query}')
    time.sleep(random.randint(5, 8))
    print(f'completed: {query}')
    return [random.randint(1, 10) for _ in range(10)]


def async_sql_queries(queries: Union[List, Tuple], max_workers: int = 3, refresh_period: int = 1):
    """make and return the output of asynchronous sql requests with timer"""
    global stop_threads
    data: List = []
    thread_timer = Thread(target=terminal_timer, args=(refresh_period,))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [executor.submit(server_request, query) for query in queries]
        thread_timer.start()
        for f in as_completed(results):
            data.append(f.result())
        stop_threads = True
    return data


# operating system #####################################################################################################

def get_size_hr(obj: Any) -> str:
    """get object size in human readable format"""
    return hr_bytes(sys.getsizeof(obj))


def py_process_memory() -> str:
    """get memory consumption of current python process in human readable format"""
    process = psutil.Process(os.getpid())
    return hr_bytes(process.memory_info().rss)


# memory & runtime tools ###############################################################################################

def show_vars(*args):
    """Print an arbitrary number of variable names with their assigned value. Iterate over all global variables until
    it finds ones with a matching memory reference."""
    pairs = {k: v for k, v in globals().items() if id(v) in map(id, args)}
    print(pairs)


def var_nm(var: Any) -> str:
    """return the variable name of any object as a string"""
    return [k for k, v in globals().items() if id(var) == id(v)][0]


if __name__ == '__main__':
    pass


# errors / tracebacks ##################################################################################################

def make_error_log() -> str:
    """if called once an exception has been raised This function returns a string error log (including type,
    msg and traceback)"""
    ex_type, ex_value, ex_traceback = sys.exc_info()
    trace_back = traceback.extract_tb(ex_traceback)
    stack_trace = [f"File : {tr[0]} , Line : {tr[1]}, Func.Name : {tr[2]}, Message : {tr[3]}" for tr in trace_back]
    stack_trace = '\n\t'.join(stack_trace)
    error_log: str = (f"Exception type: {ex_type.__name__}\n"
                      f"Exception message: {ex_value}\n"
                      f"Stack trace:\n\t {stack_trace}")
    return error_log


# emails ###############################################################################################################


def send_email(to: str, subject: str, body: str) -> None:
    """send a text email to a passed email address. the sender email address is sourced from an
    environment variable"""
    address = os.environ['EMAIL_ADDRESS']
    msg = EmailMessage()
    msg['From'] = address
    msg['To'] = to
    msg['Subject'] = subject
    msg.set_content(body)
    # msg.add_attachment(df.to_csv(index=False), filename='data.csv')
    # add a text file attachment
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(address, os.environ['EMAIL_PASSWORD'])
        smtp.send_message(msg)


def send_error_log_email(to: str) -> None:
    """
    If an exception is raised send an error log message to param email

    example
    -------

    try:
        def app():
            something

        app()
    except Exception as e:
        error_log = montk.make_error_log()
        montk.send_error_log_email(to=os.environ['EMAIL_ADDRESS'])
    """
    error_log = make_error_log()
    send_email(to=to, subject=f'Error in app: {Path(__file__).name}', body=error_log)


# monkey patching standard library classes #############################################################################

Path.ls = lambda self: list(self.iterdir())
