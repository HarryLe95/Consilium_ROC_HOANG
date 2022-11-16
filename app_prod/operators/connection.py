import utils.advancedanalytics_util as aau


class ConnectionOperator:

    @staticmethod
    def connect_(info):
        """
        Standard connect function - may end up moving into util
        """
        connection_type = None
        if info['connection_type'] == 's3':
            connection_type = aau.S3(info)
        elif info['connection_type'] == 'p2':
            connection_type = aau.P2(info)
        elif info['connection_type'] == 'file':
            connection_type = aau.File(info)
        elif info['connection_type'] == 'rts':
            connection_type = aau.RTS(info)
        elif info['connection_type'] == 'ora':
            connection_type = aau.Oracle(info)
        elif info['connection_type'] == 'sql':
            connection_type = aau.SQL(info)
        return connection_type
