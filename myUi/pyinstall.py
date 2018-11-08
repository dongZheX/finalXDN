import sys
if __name__ == '__main__':
    from PyInstaller.__main__ import run
    sys.setrecursionlimit(1000000)
    opts=['ui.py','-w','--icon=1.ico']
    run(opts)
