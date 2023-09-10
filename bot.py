from aiogram import Bot, Dispatcher, executor
from aiogram.types import Message, ContentTypes
from aiogram.dispatcher.filters import BoundFilter

from datetime import datetime
from random import randint

from logger import setup_logger
from face_recognizer.face_recognizer import FaceRecognizer


class CaptionFilter(BoundFilter):
    key = 'caption'

    def __init__(self, caption):
        self.caption = caption

    async def check(self, msg: Message):
        return len(msg.photo) and msg.caption.startswith(self.caption)

    def __iter__(self):
        yield self
        

API_TOKEN = '964269014:AAEibuPO8xZMS-Jh1nXNBNPi6ntM40nlWvs'
bot = Bot(API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def start(msg: Message):
    await msg.reply('Hello!!!')


@dp.message_handler(CaptionFilter('remember'), content_types=ContentTypes.PHOTO)
async def remember_by_photo(msg: Message):
    if not len(msg.photo):
        return
    
    text = msg.caption
    th = len('remember ')
    if len(text) <= th:
        await msg.reply('Send photo with caption like "remember Name"')
        return
    
    name = text[th:].strip()   
    photo = msg.photo[-1]
    
    # saving
    res = await photo.download(destination_dir='cache/')
    fn = res.name
    
    # detecting & rememberings
    dct = {
        name: [fn]
    }
    recognizer.remember_many(dct)
    await msg.reply('I have remember you!')


@dp.message_handler(CaptionFilter('recognize'), content_types=ContentTypes.PHOTO)
async def handle_image(msg: Message):
    if not len(msg.photo):
        return
    
    photo = msg.photo[-1]
    
    # saving
    res = await photo.download(destination_dir='cache/')

    fn = res.name
    rnd = randint(10**5, 10**6)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = hash(f'{time}_{rnd}')
    save_path = f'cache/processed/{filename}.jpg'
    
    # detecting & recognizing
    recognizer.recognize_from_image(fn, save_path)

    with open(save_path, 'rb') as img:
        await msg.reply_photo(img)

    
def handle():
    executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    WEIGHTS_PATH = 'data/faces/FaceVAE_weights.saved'
    HIDDEN_SIZE = 50

    face_rec_logger = setup_logger('face_recognizer')
    recognizer = FaceRecognizer(
        WEIGHTS_PATH, 
        HIDDEN_SIZE,
        face_rec_logger,
        )

    handle()
