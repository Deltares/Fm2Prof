# Import from standard library
import re

# Import from dependencies
import click

# Import from package
from fm2prof.main import Fm2ProfRunner
import fm2prof.utils as fmu
from fm2prof.IniFile import IniFile



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-f', help='Load configuration file')
@click.option('-n', help='Create empty ini file in current with name')
@click.option('-r', help='Run program (always use in combination with 0f)', is_flag=True)
def __cli(**kwargs):
    project = None
    if kwargs.get('f'):
        project = Fm2ProfRunner.Fm2ProfRunner(kwargs.get('f'))
    if kwargs.get('n'):
        # Create empty project based on inifile
        inifile = IniFile().print_configuration()
        new_name = kwargs.get('n')
        ini_path = f"{new_name}.ini"
        #inifile = re.sub('(NoName)', inifile, new_name)
        with open(ini_path, 'w') as f:
            f.write(inifile)
        click.echo(f"{ini_path} written to file")
    if kwargs.get('r'):
    	click.echo(project.run())
    else:
        with click.Context(__cli) as ctx:
            click.echo(ctx.get_help())

if __name__ == "__main__":
    __cli()