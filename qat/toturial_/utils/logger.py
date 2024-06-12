import logging
from datetime import datetime
import os

class ColoredFormatter(logging.Formatter):
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[94m',    # 蓝色
        'INFO': '\033[92m',     # 绿色
        'WARNING': '\033[93m',  # 黄色
        'ERROR': '\033[91m',    # 红色
        'CRITICAL': '\033[95m', # 紫红色
    }
    RESET = '\033[0m'  # 重置颜色

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        log_message = super().format(record)
        return f'{log_color}{log_message}{self.RESET}'

def set_logger(console=True,logging_level=logging.INFO,save_dir='.'):

    save_file_prefix= 'log_'
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H%M')

    if not os.path.exists(os.path.join(save_dir, 'logs')):
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
    save_file = os.path.join(save_dir, 'logs', save_file_prefix + timestamp + '.log' )

    logging.basicConfig(level=logging_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
                        filename=save_file,  # 设置日志文件路径
                        encoding='utf-8',
                        filemode='w'  # 以写模式打开日志文件
    )
    # 创建日志记录器
    logger = logging.getLogger()
    print('logger save_file:', save_file)
    if console:
        console_handler =logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter= ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.info('日志记录器已开启')
    return logger

if __name__ == '__main__':
    logger=set_logger(save_dir=os.getcwd()+r'\qat\toturial')
    logger.info('hello world')