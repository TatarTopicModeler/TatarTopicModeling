import re

TT_ALPHA_REGEX = r'аәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя'

sound_pairs = {'б': 'п', 'в': 'ф', 'г': 'к', 'д': 'т', 'ж': 'ш', 'җ': 'щ', 'з': 'с'}
deaf_pairs = {v: k for k, v in sound_pairs.items()}


def sound_deaf(intext, m1='', m2='_'):
    text = intext
    for k, v in sound_pairs.items():
        text = re.sub(k + m1 + r'\b', v + m2, text)
    for k, v in deaf_pairs.items():
        text = re.sub(k + m1 + r'\b', v + m2, text)

    return text


def reduce_affix(text, affixes, m1='', m2='_'):
    #     regex = '|'.join(affixes)
    #     regex = f'({regex})'
    #     regex = regex+r'\b'
    reduced = text
    for a in affixes:
        reduced = re.sub(a + m1 + r'\b', m2, reduced)

    return reduced


figil_shart_barlik_affix = 'сам саң са сак сагыз салар'.split()
figil_shart_yuklik_affix = ['ма' + a for a in figil_shart_barlik_affix]
figil_past_definite_affix = 'дым дем дың дең ды де дык дек дылар деләр дыгыз дегез'.split()
figil_past_indefinite_affix = 'ганмын гәнмен ганбыз гәнбез гансың гәнсең гансыз гәнсез ган гән ганнар гәннәр'.split()
figil_present_affix = 'на нә м мын мен быз без сың сең сыз сез лар ләр нәр нәр'.split()
figil_future_affix = 'ыр ер ар әр'


def process_figil(intext):
    text = intext
    text = reduce_affix(text, figil_past_definite_affix)
    text = reduce_affix(text, figil_past_indefinite_affix)
    text = reduce_affix(text, figil_shart_yuklik_affix)
    text = reduce_affix(text, figil_shart_barlik_affix)
    text = reduce_affix(text, figil_present_affix)
    return text


ravesh_affix = 'ып еп'.split()


def process_ravesh(intext):
    text = intext
    text = reduce_affix(text, ravesh_affix, '', '_')
    return text


# ISEM
kilesh = 'ның нең га гә ка кә ны не дан дән тан тән нда ндә нта нтә да дә та тә'.split()
tartim = 'ен ым ем ың ең ы е м ң сы се ыбыз ебез ыгыз егез быз без гыз гез'.split()
tartim = sorted(tartim, key=lambda x: len(x), reverse=True)
san = 'лар ләр нар нәр'.split()


def process_isem(intext, deaf=False, verbose=False):
    text = intext
    text = reduce_affix(text, kilesh, m1='', m2='K')
    if verbose: print('k', text)
    text = reduce_affix(text, tartim, m1='K', m2='T')
    if verbose: print('t', text)
    text = reduce_affix(text, tartim, m1='', m2='T')
    if verbose: print('t', text)
    if deaf:
        text = sound_deaf(text, m1='T', m2='_')

    text = reduce_affix(text, san, m1='T', m2='_')
    text = re.sub('[KT]', '_', text)
    return text


# SIYFAT
comparative_affix = 'рак рәк'.split()
tartim_siyfat = 'гы ге'.split()


def process_siyfat(intext, verbose=False):
    text = intext
    text = reduce_affix(text, comparative_affix)
    if verbose: print('c', text)
    text = reduce_affix(text, tartim_siyfat)
    if verbose: print('t', text)
    return text


def tatar_stemmer(text):
    text = process_ravesh(text)
    text = process_siyfat(text)
    text = process_isem(text, deaf=False)
    text = process_figil(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'_+', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == '__main__':
    text = 'бавырсакка татарларга миллинең ризыгы камыр кисәкләрен кайнаган майда ' + \
           'кыздырып әзерләнә әзерләү ысулы савытка йомырка сытсалар сыталар сөт салалар май комы өстиләр ' + \
           'бардым эшләдем бардык эшләдек бардың эшләдең бардыгыз эшләдегез барды эшләде бардылар эшләделәр ' + \
           ' '

    # text = 'баручыларның китабым әбием кисәгем'
    text = 'татарстан татарлар өлкәсендә инглиз империясе республикасы өлеше рәисе президенты хәзерге'
    text = 'өлкәсендә тамагында'
    text = 'хәзерге борынгы'
    text = 'тәрҗемәләре'
    print(text)
    print(tatar_stemmer(text))
