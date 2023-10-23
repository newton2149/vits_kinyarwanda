""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!|?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ'͡'"


_kinyarwanda_letters = "ɑɛɪɔʌŋʀʁ"
_kinyarwanda_pronunciation = "áéíóúɑɛɪɔʌ"


_data_letters_ipa = "̃"

# # Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_data_letters_ipa) + list(_kinyarwanda_letters) + list(_kinyarwanda_pronunciation)

# # Special symbol ids
SPACE_ID = symbols.index(" ")
