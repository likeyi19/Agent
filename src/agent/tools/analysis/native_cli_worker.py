"""MACS3's installed argparse interface, executed as argv without a shell."""
import hashlib
import argparse
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True


def parser(executable):
    return runpy.run_path(executable, run_name='agent_native_metadata')['prepare_argparser']()


def subcommands(executable):
    root = parser(executable)
    action = next(a for a in root._actions if isinstance(a, argparse._SubParsersAction))
    return action.choices


def metadata(name, command):
    parameters = []
    for action in command._actions:
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction)): continue
        default = action.default
        try: json.dumps(default, allow_nan=False); known = True
        except (TypeError, ValueError): known = False
        boolean = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
        kind = 'boolean' if boolean else 'list' if action.nargs in ('+', '*') or isinstance(action.nargs, int) else 'integer' if action.type is int else 'number' if action.type is float else 'string'
        parameters.append(dict(name=action.dest, kind=kind, annotation='', parameter_kind='POSITIONAL_OR_KEYWORD',
            required=action.required, default_known=known, default=default if known else None,
            default_label=str(default) if known else 'Native default', choices=list(action.choices) if action.choices else [],
            options=action.option_strings, help=action.help,
            input_kind='file' if action.dest in ('ifile','tfile','cfile','t1','c1','t2','c2','input_file') else None))
    return dict(name=name, signature=command.format_usage().strip(), parameters=parameters,
        description=command.description or '', limitation=None)


def main():
    if sys.argv[1]=='--discover':
        exe=sys.argv[2]; selected=(sys.argv[3] or None) if len(sys.argv)>3 else None
        found=subcommands(exe)
        encoded=json.dumps(dict(package='MACS3',version=runpy.run_path(exe,run_name='agent_native_metadata')['MACS_VERSION'],
            functions=[metadata(k,v) if selected else dict(name=k) for k,v in found.items() if selected is None or k==selected]))
        if len(sys.argv)>4: Path(sys.argv[4]).write_text(encoded)
        else: print(encoded)
        return
    request=json.loads(Path(sys.argv[1]).read_text()); root=Path(request['stage'])
    work=root/'work';work.mkdir();objects=root/'objects';objects.mkdir()
    def write(name,value): (root/name).write_text(json.dumps(value,allow_nan=False))
    succeeded=False
    try:
        exe=request['executable'];name=request['function'];command=subcommands(exe)[name]
        actions={a.dest:a for a in command._actions}
        supplied=dict(request['parameters'])
        for key,binding in request['inputs'].items():
            if binding['adapter']=='accepted' and binding['descriptor']['encoding']=='file':
                data=json.loads(Path(binding['result_path']).read_text())
                index=int(binding['selection'].removeprefix('file:'))
                path=Path(binding['root'])/data['produced_files'][index]['file']
            elif binding['adapter']=='file': path=Path(binding['path'])
            else: raise ValueError('CLI input requires a managed file representation')
            destination=work/('input-'+str(len(list(work.iterdir())))+path.suffix)
            shutil.copyfile(path,destination);supplied[key]=str(destination)
        for key,name_ in request['outputs'].items(): supplied[key]=str(work/name_)
        argv=[exe,name]
        for key,value in supplied.items():
            action=actions[key]
            flag=next((v for v in action.option_strings if v.startswith('--')),action.option_strings[0] if action.option_strings else None)
            if isinstance(action,(argparse._StoreTrueAction,argparse._StoreFalseAction)):
                if type(value) is not bool: raise ValueError('CLI switch requires a boolean')
                if value is action.const: argv.append(flag)
                elif value != action.default: raise ValueError('This CLI switch cannot express the selected boolean')
                continue
            if value is None: raise ValueError('This CLI argument has no explicit null representation')
            values=value if isinstance(value,list) else [value]
            if not flag: argv.extend(map(str,values))
            elif len(values)==1 and flag.startswith('--'): argv.append(flag+'='+str(values[0]))
            else:
                if any(str(v).startswith('-') and not isinstance(v,(int,float)) for v in values):
                    raise ValueError('CLI vector values cannot introduce option flags')
                argv.append(flag);argv.extend(map(str,values))
        def checksum(path):
            with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
        before={p:checksum(p) for p in work.rglob('*') if p.is_file()}
        with (root/'stdout.txt').open('wb') as stdout, (root/'stderr.txt').open('wb') as stderr:
            completed=subprocess.run(argv,cwd=work,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL)
        if completed.returncode: raise ValueError((root/'stderr.txt').read_text()[-8000:] or 'Native CLI exited with '+str(completed.returncode))
        succeeded=True
        receipt=dict(native_execution='succeeded',native_exit_code=completed.returncode,package='MACS3',function_name=name,
            version=runpy.run_path(exe,run_name='agent_native_metadata')['MACS_VERSION'],
            native_output_type='CLI.no_return_value',supplied_parameters=request['parameters'])
        write('execution.json',receipt)
        produced=[]
        for path in sorted(work.rglob('*')):
            if path.is_dir(): continue
            if before.get(path) == checksum(path): continue
            if path.is_symlink(): raise ValueError('Native-produced symbolic links are not portable outputs')
            target=objects/('file-'+str(len(produced))+path.suffix);shutil.copyfile(path,target)
            produced.append(dict(name=str(path.relative_to(work)),file=target.name))
        if (root/'stdout.txt').stat().st_size:
            shutil.copyfile(root/'stdout.txt',objects/'stdout.txt');produced.append(dict(name='Standard output',file='stdout.txt'))
        result=dict(return_value=dict(type='CLI.no_return_value',encoding='json',value=None),modified_inputs={},produced_files=produced)
        write('result.json',result);write('preview.json',result)
    except Exception as exc:
        write('error.json',dict(message=str(exc),type=type(exc).__name__,native_execution='succeeded' if succeeded else 'failed',result_representation='failed' if succeeded else 'not_started'))
        sys.exit(1)


if __name__=='__main__': main()
