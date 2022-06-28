import configparser
import Logger
import datetime


class ConfigWrapper():
    def __init__(self):
        self.config_parser = configparser.ConfigParser()
        self.logger_initiated = False

    def read(self, path):
        self.config_parser.read(path)

    def get(self, section, option):
        value = self.config_parser.get(section, option)
        if self.logger_initiated:
            Logger.debug("[Config] [{0}] \'{1}\' = {2}".format(section, option, value))
        return value

    def getboolean(self, section, option):
        value = self.config_parser.getboolean(section, option)
        if self.logger_initiated:
            Logger.debug("[Config] [{0}] \'{1}\' = {2}".format(section, option, value))
        return value

    def getint(self, section, option):
        value = self.config_parser.getint(section, option)
        if self.logger_initiated:
            Logger.debug("[Config] [{0}] \'{1}\' = {2}".format(section, option, value))
        return value

    def getfloat(self, section, option):
        value = self.config_parser.getfloat(section, option)
        if self.logger_initiated:
            Logger.debug("[Config] [{0}] \'{1}\' = {2}".format(section, option, value))
        return value

    # my helper funcs
    def get_list_as_int(self, section, option):
        value = self.config_parser.get(section, option)
        if self.logger_initiated:
            Logger.debug("[Config] [{0}] \'{1}\' = {2}".format(section, option, value))

        list_input = value.strip('[]').replace(' ', '').split(',')
        try:
            int_list_out = [int(x) for x in list_input]
        except ValueError as e:
            Logger.log("[Config ERROR] Could not convert string to int list!", Logger.LogLevel.quiet)
            raise e
        return int_list_out

    def get_list_as_string(self, section, option):
        value = self.config_parser.get(section, option)
        if self.logger_initiated:
            Logger.debug("[Config] [{0}] \'{1}\' = {2}".format(section, option, value))

        list_input = value.strip('[]').replace(' ', '').split(',')
        return list_input
            
    def get_list_as_float(self, section, option):
        value = self.config_parser.get(section, option)
        if self.logger_initiated:
            Logger.debug("[Config] [{0}] \'{1}\' = {2}".format(section, option, value))

        list_input = value.strip('[]').replace(' ', '').split(',')
        try:
            float_list_out = [float(x) for x in list_input]
        except ValueError as e:
            Logger.log("[Config ERROR] Could not convert string to int list!", Logger.LogLevel.quiet)
            raise e
        return float_list_out

    def get_list_as_datetimes(self, section, option):
        value = self.config_parser.get(section, option)
        if self.logger_initiated:
            Logger.debug("[Config] [{0}] \'{1}\' = {2}".format(section, option, value))

        list_input = value.strip('[]').replace(' ', '').replace('\'','').split(',')
        if "None" in list_input:
            return None
        try:
            float_list_out = [datetime.datetime.strptime(x, "%Y-%m-%d_%H-%M-%S") for x in list_input if x and x.strip()]
        except ValueError as e:
            Logger.log("[Config ERROR] Could not convert string to int list!", Logger.LogLevel.quiet)
            raise e
        return float_list_out

    def parse_section_as_dict(self, section, type='float'):
        dict_keys = self.config_parser.options(section)
        section_dict = {}
        for key in dict_keys:
            if type=='float':
                section_dict[key] = self.getfloat(section, key)
            elif type == 'bool':
                section_dict[key] = self.getboolean(section, key)
            elif type == 'int':
                section_dict[key] = self.getint(section, key)
            else:
                section_dict[key] = self.get(section, key)
        return section_dict