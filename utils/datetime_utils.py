from datetime import datetime


def time_delta(t: datetime):
    d = datetime.now()-t
    hours = d.seconds//3600
    minutes = (d.seconds-hours*3600) // 60
    seconds = d.seconds % 60
    return f'{hours:02}:{minutes:02}:{seconds:02}'

