from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="BOT",
    settings_files=['settings.toml', '.secrets.toml'],
    environments=True,
    load_dotenv=True,
    env_switcher="BOT_ENV",
    validators=[
        Validator("DEBUG", must_exist=True)
    ]
)


# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
