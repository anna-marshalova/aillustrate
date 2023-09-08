import asyncio
from aiogram import Bot, Dispatcher
from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import Command
from pipeline import Pipeline
import time
from aiogram.types import FSInputFile

pipeline = Pipeline()
router = Router()

@router.message(Command('start'))
async def cmd_start(message: Message):
    return await message.reply('Привет, отправь мне любой текст, и я отвечу тебе изображением')

@router.message(F.text)
async def cmd_start(message: Message):
    await message.reply('Пожалуйста, подождите, ваше изображение будет создано в течение 20-30 секунд!')
    name = str(message.from_user.id)+"_"+str(time.time()) 
    image = pipeline.generate(message.text)
    image.save('images/'+name+'.png')
    img = FSInputFile('images/'+name+'.png')
    await message.reply_photo(img)
    
    # return await message.reply_sticker('CAACAgIAAx0CaJZtbgACEqVkh3SMPMgap7qMSc33DGHhTuUZ3AACXCgAAlpvGUgMxewk-BAF-i8E')

async def main() -> None:
    bot = Bot(
        token=r"6650206534:AAFUiGNlfN5_BTnGYYIBp4eSZHzkQ2mVOWc"
    )
    dp = Dispatcher()
    dp.include_routers(
        router,
    )
    print('___START____')
    await dp.start_polling(
        bot, 
        allowed_updates=dp.resolve_used_update_types(),
    )

if __name__ == "__main__":
    asyncio.run(main())