import asyncio
import time

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message

from pipeline import Pipeline

pipeline = Pipeline()
router = Router()
TOKEN = r""

@router.message(Command('start'))
async def cmd_start(message: Message):
    return await message.reply('🙋‍♀️Привет, отправь мне любой текст, и я отвечу тебе изображением')

@router.message(F.text)
async def any_text(message: Message):
    await message.reply('👩‍🎨Пожалуйста, подождите, ваше изображение будет создано в течение 5-10 секунд!')
    name = str(message.from_user.id)+"_"+str(time.time()) 
    image = pipeline.generate(message.text)
    image.save('images/'+name+'.png')
    img = FSInputFile('images/'+name+'.png')
    await message.reply_photo(img)

async def main() -> None:
    bot = Bot(token=TOKEN)
    dp = Dispatcher()
    dp.include_routers(router)
    print('____START____')
    await dp.start_polling(
        bot, 
        allowed_updates=dp.resolve_used_update_types(),
    )

if __name__ == "__main__":
    asyncio.run(main())