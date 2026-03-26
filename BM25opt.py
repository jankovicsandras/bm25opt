################################################################################
#
#  BM25opt : faster BM25 search algorithms in Python
#  version 1.2.0
#  by András Jankovics and contributors  https://github.com/jankovicsandras  andras@jankovics.net
#  based on https://github.com/dorianbrown/rank_bm25 by Dorian Brown
#  Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0
#
################################################################################


import math, numpy as np


def stop_words_filter(lang):
  stopwords = {
    "en": {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "would", "should", "could", "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't", "let's", "that's", "who's", "what's", "here's", "there's", "when's", "where's", "why's", "how's", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"},
    "fr": {"au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux", "il", "je", "la", "le", "leur", "lui", "ma", "mais", "me", "même", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "c", "d", "j", "l", "à", "m", "n", "s", "t", "y", "étée", "étées", "étant", "suis", "es", "êtes", "sont", "serai", "seras", "sera", "serons", "serez", "seront", "serais", "serait", "serions", "seriez", "seraient", "étais", "était", "étions", "étiez", "étaient", "fus", "fut", "fûmes", "fûtes", "furent", "sois", "soit", "soyons", "soyez", "soient", "fusse", "fusses", "fussions", "fussiez", "fussent", "ayant", "eu", "eue", "eues", "eus", "ai", "avons", "avez", "ont", "aurai", "aurons", "aurez", "auront", "aurais", "aurait", "aurions", "auriez", "auraient", "avais", "avait", "aviez", "avaient", "eut", "eûmes", "eûtes", "eurent", "aie", "aies", "ait", "ayons", "ayez", "aient", "eusse", "eusses", "eût", "eussions", "eussiez", "eussent", "ceci", "cela", "celà", "cet", "cette", "ici", "ils", "les", "leurs", "quel", "quels", "quelle", "quelles", "sans", "soi"},
    "es": {"de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están", "esté", "estés", "estemos", "estéis", "estén", "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán", "estaría", "estarías", "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos", "estabais", "estaban", "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron", "estuviera", "estuvieras", "estuviéramos", "estuvierais", "estuvieran", "estuviese", "estuvieses", "estuviésemos", "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados", "estadas", "estad", "he", "has", "ha", "hemos", "habéis", "han", "haya", "hayas", "hayamos", "hayáis", "hayan", "habré", "habrás", "habrá", "habremos", "habréis", "habrán", "habría", "habrías", "habríamos", "habríais", "habrían", "había", "habías", "habíamos", "habíais", "habían", "hube", "hubiste", "hubo", "hubimos", "hubisteis", "hubieron", "hubiera", "hubieras", "hubiéramos", "hubierais", "hubieran", "hubiese", "hubieses", "hubiésemos", "hubieseis", "hubiesen", "habiendo", "habido", "habida", "habidos", "habidas", "soy", "eres", "es", "somos", "sois", "son", "sea", "seas", "seamos", "seáis", "sean", "seré", "serás", "será", "seremos", "seréis", "serán", "sería", "serías", "seríamos", "seríais", "serían", "era", "eras", "éramos", "erais", "eran", "fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron", "fuera", "fueras", "fuéramos", "fuerais", "fueran", "fuese", "fueses", "fuésemos", "fueseis", "fuesen", "siendo", "sido", "tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen", "tenga", "tengas", "tengamos", "tengáis", "tengan", "tendré", "tendrás", "tendrá", "tendremos", "tendréis", "tendrán", "tendría", "tendrías", "tendríamos", "tendríais", "tendrían", "tenía", "tenías", "teníamos", "teníais", "tenían", "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron", "tuviera", "tuvieras", "tuviéramos", "tuvierais", "tuvieran", "tuviese", "tuvieses", "tuviésemos", "tuvieseis", "tuviesen", "teniendo", "tenido", "tenida", "tenidos", "tenidas", "tened"},
    "pt": {"de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "ao", "ele", "das", "à", "seu", "sua", "ou", "quando", "muito", "nos", "já", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre", "depois", "sem", "mesmo", "aos", "seus", "quem", "nas", "me", "esse", "eles", "você", "essa", "num", "nem", "suas", "meu", "às", "minha", "numa", "pelos", "elas", "qual", "nós", "lhe", "deles", "essas", "esses", "pelas", "este", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo", "estou", "está", "estamos", "estão", "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos", "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam", "estivesse", "estivéssemos", "estivessem", "estiver", "estivermos", "estiverem", "hei", "há", "havemos", "hão", "houve", "houvemos", "houveram", "houvera", "houvéramos", "haja", "hajamos", "hajam", "houvesse", "houvéssemos", "houvessem", "houver", "houvermos", "houverem", "houverei", "houverá", "houveremos", "houverão", "houveria", "houveríamos", "houveriam", "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi", "fomos", "foram", "fora", "fôramos", "seja", "sejamos", "sejam", "fosse", "fôssemos", "fossem", "for", "formos", "forem", "serei", "será", "seremos", "serão", "seria", "seríamos", "seriam", "tenho", "tem", "temos", "tém", "tinha", "tínhamos", "tinham", "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos", "tenha", "tenhamos", "tenham", "tivesse", "tivéssemos", "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá", "teremos", "terão", "teria", "teríamos", "teriam"},
    "it": {"ad", "al", "allo", "ai", "agli", "all", "agl", "alla", "alle", "con", "col", "coi", "da", "dal", "dallo", "dai", "dagli", "dall", "dagl", "dalla", "dalle", "di", "del", "dello", "dei", "degli", "dell", "degl", "della", "delle", "in", "nel", "nello", "nei", "negli", "nell", "negl", "nella", "nelle", "su", "sul", "sullo", "sui", "sugli", "sull", "sugl", "sulla", "sulle", "per", "tra", "contro", "io", "tu", "lui", "lei", "noi", "voi", "loro", "mio", "mia", "miei", "mie", "tuo", "tua", "tuoi", "tue", "suo", "sua", "suoi", "sue", "nostro", "nostra", "nostri", "nostre", "vostro", "vostra", "vostri", "vostre", "mi", "ti", "ci", "vi", "lo", "la", "li", "le", "gli", "ne", "il", "un", "uno", "una", "ma", "ed", "se", "perché", "anche", "come", "dov", "dove", "che", "chi", "cui", "non", "più", "quale", "quanto", "quanti", "quanta", "quante", "quello", "quelli", "quella", "quelle", "questo", "questi", "questa", "queste", "si", "tutto", "tutti", "a", "c", "e", "i", "l", "o", "ho", "hai", "ha", "abbiamo", "avete", "hanno", "abbia", "abbiate", "abbiano", "avrò", "avrai", "avrà", "avremo", "avrete", "avranno", "avrei", "avresti", "avrebbe", "avremmo", "avreste", "avrebbero", "avevo", "avevi", "aveva", "avevamo", "avevate", "avevano", "ebbi", "avesti", "ebbe", "avemmo", "aveste", "ebbero", "avessi", "avesse", "avessimo", "avessero", "avendo", "avuto", "avuta", "avuti", "avute", "sono", "sei", "è", "siamo", "siete", "sia", "siate", "siano", "sarò", "sarai", "sarà", "saremo", "sarete", "saranno", "sarei", "saresti", "sarebbe", "saremmo", "sareste", "sarebbero", "ero", "eri", "era", "eravamo", "eravate", "erano", "fui", "fosti", "fu", "fummo", "foste", "furono", "fossi", "fosse", "fossimo", "fossero", "essendo", "faccio", "fai", "facciamo", "fanno", "faccia", "facciate", "facciano", "farò", "farai", "farà", "faremo", "farete", "faranno", "farei", "faresti", "farebbe", "faremmo", "fareste", "farebbero", "facevo", "facevi", "faceva", "facevamo", "facevate", "facevano", "feci", "facesti", "fece", "facemmo", "faceste", "fecero", "facessi", "facesse", "facessimo", "facessero", "facendo", "sto", "stai", "sta", "stiamo", "stanno", "stia", "stiate", "stiano", "starò", "starai", "starà", "staremo", "starete", "staranno", "starei", "staresti", "starebbe", "staremmo", "stareste", "starebbero", "stavo", "stavi", "stava", "stavamo", "stavate", "stavano", "stetti", "stesti", "stette", "stemmo", "steste", "stettero", "stessi", "stesse", "stessimo", "stessero", "stando"},
    "de": {"aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander", "andere", "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders", "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da", "damit", "dann", "der", "den", "des", "dem", "die", "das", "daß", "derselbe", "derselben", "denselben", "desselben", "demselben", "dieselbe", "dieselben", "dasselbe", "dazu", "dein", "deine", "deinem", "deinen", "deiner", "deines", "denn", "derer", "dessen", "dich", "dir", "du", "dies", "diese", "diesem", "diesen", "dieser", "dieses", "doch", "dort", "durch", "ein", "eine", "einem", "einen", "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger", "einiges", "einmal", "er", "ihn", "ihm", "es", "etwas", "euer", "eure", "eurem", "euren", "eurer", "eures", "für", "gegen", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten", "hier", "hin", "hinter", "ich", "mich", "mir", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres", "euch", "im", "in", "indem", "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jene", "jenem", "jenen", "jener", "jenes", "jetzt", "kann", "kein", "keine", "keinem", "keinen", "keiner", "keines", "können", "könnte", "machen", "man", "manche", "manchem", "manchen", "mancher", "manches", "mein", "meine", "meinem", "meinen", "meiner", "meines", "mit", "muss", "musste", "nach", "nicht", "nichts", "noch", "nun", "nur", "ob", "oder", "ohne", "sehr", "sein", "seine", "seinem", "seinen", "seiner", "seines", "selbst", "sich", "sie", "ihnen", "sind", "so", "solche", "solchem", "solchen", "solcher", "solches", "soll", "sollte", "sondern", "sonst", "über", "um", "und", "uns", "unse", "unsem", "unsen", "unser", "unses", "unter", "viel", "vom", "von", "vor", "während", "war", "waren", "warst", "was", "weg", "weil", "weiter", "welche", "welchem", "welchen", "welcher", "welches", "wenn", "werde", "werden", "wie", "wieder", "will", "wir", "wird", "wirst", "wo", "wollen", "wollte", "würde", "würden", "zu", "zum", "zur", "zwar", "zwischen"},
    "nl": {"de", "en", "van", "ik", "te", "dat", "die", "in", "een", "hij", "het", "niet", "zijn", "is", "was", "op", "aan", "met", "als", "voor", "had", "er", "maar", "om", "hem", "dan", "zou", "of", "wat", "mijn", "men", "dit", "zo", "door", "over", "ze", "zich", "bij", "ook", "tot", "je", "mij", "uit", "der", "daar", "haar", "naar", "heb", "hoe", "heeft", "hebben", "deze", "u", "want", "nog", "zal", "me", "zij", "nu", "ge", "geen", "omdat", "iets", "worden", "toch", "al", "waren", "veel", "meer", "doen", "toen", "moet", "ben", "zonder", "kan", "hun", "dus", "alles", "onder", "ja", "eens", "hier", "wie", "werd", "altijd", "doch", "wordt", "wezen", "kunnen", "ons", "zelf", "tegen", "na", "reeds", "wil", "kon", "niets", "uw", "iemand", "geweest", "andere"},
    "sv": {"och", "det", "att", "i", "en", "jag", "hon", "som", "han", "på", "den", "med", "var", "sig", "för", "så", "till", "är", "men", "ett", "om", "hade", "de", "av", "icke", "mig", "du", "henne", "då", "sin", "nu", "har", "inte", "hans", "honom", "skulle", "hennes", "där", "min", "man", "ej", "vid", "kunde", "något", "från", "ut", "när", "efter", "upp", "vi", "dem", "vara", "vad", "över", "än", "dig", "kan", "sina", "här", "ha", "mot", "alla", "under", "någon", "eller", "allt", "mycket", "sedan", "ju", "denna", "själv", "detta", "åt", "utan", "varit", "hur", "ingen", "mitt", "ni", "bli", "blev", "oss", "din", "dessa", "några", "deras", "blir", "mina", "samma", "vilken", "er", "sådan", "vår", "blivit", "dess", "inom", "mellan", "sådant", "varför", "varje", "vilka", "ditt", "vem", "vilket", "sitt", "sådana", "vart", "dina", "vars", "vårt", "våra", "ert", "era", "vilkas"},
    "no": {"og", "i", "jeg", "det", "at", "en", "et", "den", "til", "er", "som", "på", "de", "med", "han", "av", "ikke", "ikkje", "der", "så", "var", "meg", "seg", "men", "ett", "har", "om", "vi", "min", "mitt", "ha", "hadde", "hun", "nå", "over", "da", "ved", "fra", "du", "ut", "sin", "dem", "oss", "opp", "man", "kan", "hans", "hvor", "eller", "hva", "skal", "selv", "sjøl", "her", "alle", "vil", "bli", "ble", "blei", "blitt", "kunne", "inn", "når", "være", "kom", "noen", "noe", "ville", "dere", "deres", "kun", "ja", "etter", "ned", "skulle", "denne", "for", "deg", "si", "sine", "sitt", "mot", "å", "meget", "hvorfor", "dette", "disse", "uten", "hvordan", "ingen", "din", "ditt", "blir", "samme", "hvilken", "hvilke", "sånn", "inni", "mellom", "vår", "hver", "hvem", "vors", "hvis", "både", "bare", "enn", "fordi", "før", "mange", "også", "slik", "vært", "båe", "begge", "siden", "dykk", "dykkar", "dei", "deira", "deires", "deim", "di", "då", "eg", "ein", "eit", "eitt", "elles", "honom", "hjå", "ho", "hoe", "henne", "hennar", "hennes", "hoss", "hossen", "ingi", "inkje", "korleis", "korso", "kva", "kvar", "kvarhelst", "kven", "kvi", "kvifor", "me", "medan", "mi", "mine", "mykje", "no", "nokon", "noka", "nokor", "noko", "nokre", "sia", "sidan", "so", "somt", "somme", "um", "upp", "vere", "vore", "verte", "vort", "varte", "vart"},
    "da": {"og", "i", "jeg", "det", "at", "en", "den", "til", "er", "som", "på", "de", "med", "han", "af", "for", "ikke", "der", "var", "mig", "sig", "men", "et", "har", "om", "vi", "min", "havde", "ham", "hun", "nu", "over", "da", "fra", "du", "ud", "sin", "dem", "os", "op", "man", "hans", "hvor", "eller", "hvad", "skal", "selv", "her", "alle", "vil", "blev", "kunne", "ind", "når", "være", "dog", "noget", "ville", "jo", "deres", "efter", "ned", "skulle", "denne", "end", "dette", "mit", "også", "under", "have", "dig", "anden", "hende", "mine", "alt", "meget", "sit", "sine", "vor", "mod", "disse", "hvis", "din", "nogle", "hos", "blive", "mange", "ad", "bliver", "hendes", "været", "thi", "jer", "sådan"},
    "ru": {"и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "сказал", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "человек", "чего", "раз", "тоже", "себе", "под", "жизнь", "будет", "ж", "тогда", "кто", "этот", "говорил", "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "кажется", "сейчас", "были", "куда", "зачем", "сказать", "всех", "никогда", "сегодня", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "сказала", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между"},
    "fi": {"olla", "olen", "olet", "on", "olemme", "olette", "ovat", "ole", "oli", "olisi", "olisit", "olisin", "olisimme", "olisitte", "olisivat", "olit", "olin", "olimme", "olitte", "olivat", "ollut", "olleet", "en", "et", "ei", "emme", "ette", "eivät", "minä", "minun", "minut", "minua", "minussa", "minusta", "minuun", "minulla", "minulta", "minulle", "sinä", "sinun", "sinut", "sinua", "sinussa", "sinusta", "sinuun", "sinulla", "sinulta", "sinulle", "hän", "hänen", "hänet", "häntä", "hänessä", "hänestä", "häneen", "hänellä", "häneltä", "hänelle", "me", "meidän", "meidät", "meitä", "meissä", "meistä", "meihin", "meillä", "meiltä", "meille", "te", "teidän", "teidät", "teitä", "teissä", "teistä", "teihin", "teillä", "teiltä", "teille", "he", "heidän", "heidät", "heitä", "heissä", "heistä", "heihin", "heillä", "heiltä", "heille", "tämä", "tämän", "tätä", "tässä", "tästä", "tähän", "tällä", "tältä", "tälle", "tänä", "täksi", "tuo", "tuon", "tuota", "tuossa", "tuosta", "tuohon", "tuolla", "tuolta", "tuolle", "tuona", "tuoksi", "se", "sen", "sitä", "siinä", "siitä", "siihen", "sillä", "siltä", "sille", "sinä", "siksi", "nämä", "näiden", "näitä", "näissä", "näistä", "näihin", "näillä", "näiltä", "näille", "näinä", "näiksi", "nuo", "noiden", "noita", "noissa", "noista", "noihin", "noilla", "noilta", "noille", "noina", "noiksi", "ne", "niiden", "niitä", "niissä", "niistä", "niihin", "niillä", "niiltä", "niille", "niinä", "niiksi", "kuka", "kenen", "kenet", "ketä", "kenessä", "kenestä", "keneen", "kenellä", "keneltä", "kenelle", "kenenä", "keneksi", "ketkä", "keiden", "ketkä", "keitä", "keissä", "keistä", "keihin", "keillä", "keiltä", "keille", "keinä", "keiksi", "mikä", "minkä", "minkä", "mitä", "missä", "mistä", "mihin", "millä", "miltä", "mille", "minä", "miksi", "mitkä", "joka", "jonka", "jota", "jossa", "josta", "johon", "jolla", "jolta", "jolle", "jona", "joksi", "jotka", "joiden", "joita", "joissa", "joista", "joihin", "joilla", "joilta", "joille", "joina", "joiksi", "että", "ja", "jos", "koska", "kuin", "mutta", "niin", "sekä", "sillä", "tai", "vaan", "vai", "vaikka", "kanssa", "mukaan", "noin", "poikki", "yli", "kun", "nyt", "itse"},
    "hu": {"a", "ahogy", "ahol", "aki", "akik", "akkor", "alatt", "által", "általában", "amely", "amelyek", "amelyekben", "amelyeket", "amelyet", "amelynek", "ami", "amit", "amolyan", "amíg", "amikor", "át", "abban", "ahhoz", "annak", "arra", "arról", "az", "azok", "azon", "azt", "azzal", "azért", "aztán", "azután", "azonban", "bár", "be", "belül", "benne", "cikk", "cikkek", "cikkeket", "csak", "de", "e", "eddig", "egész", "egy", "egyes", "egyetlen", "egyéb", "egyik", "egyre", "ekkor", "el", "elég", "ellen", "elő", "először", "előtt", "első", "én", "éppen", "ebben", "ehhez", "emilyen", "ennek", "erre", "ez", "ezt", "ezek", "ezen", "ezzel", "ezért", "és", "fel", "felé", "hanem", "hiszen", "hogy", "hogyan", "igen", "így", "illetve", "ill.", "ill", "ilyen", "ilyenkor", "ison", "ismét", "itt", "jó", "jól", "jobban", "kell", "kellett", "keresztül", "keressünk", "ki", "kívül", "között", "közül", "legalább", "lehet", "lehetett", "legyen", "lenne", "lenni", "lesz", "lett", "maga", "magát", "majd", "majd", "már", "más", "másik", "meg", "még", "mellett", "mert", "mely", "melyek", "mi", "mit", "míg", "miért", "milyen", "mikor", "minden", "mindent", "mindenki", "mindig", "mint", "mintha", "mivel", "most", "nagy", "nagyobb", "nagyon", "ne", "néha", "nekem", "neki", "nem", "néhány", "nélkül", "nincs", "olyan", "ott", "össze", "ő", "ők", "őket", "pedig", "persze", "rá", "s", "saját", "sem", "semmi", "sok", "sokat", "sokkal", "számára", "szemben", "szerint", "szinte", "talán", "tehát", "teljes", "tovább", "továbbá", "több", "úgy", "ugyanis", "új", "újabb", "újra", "után", "utána", "utolsó", "vagy", "vagyis", "valaki", "valami", "valamint", "való", "vagyok", "van", "vannak", "volt", "voltam", "voltak", "voltunk", "vissza", "vele", "viszont", "volna"},
    "ga": {"a", "ach", "ag", "agus", "an", "aon", "ar", "arna", "as", "b'", "ba", "beirt", "bhúr", "caoga", "ceathair", "ceathrar", "chomh", "chtó", "chuig", "chun", "cois", "céad", "cúig", "cúigear", "d'", "daichead", "dar", "de", "deich", "deichniúr", "den", "dhá", "do", "don", "dtí", "dá", "dár", "dó", "faoi", "faoin", "faoina", "faoinár", "fara", "fiche", "gach", "gan", "go", "gur", "haon", "hocht", "i", "iad", "idir", "in", "ina", "ins", "inár", "is", "le", "leis", "lena", "lenár", "m'", "mar", "mo", "mé", "na", "nach", "naoi", "naonúr", "ná", "ní", "níor", "nó", "nócha", "ocht", "ochtar", "os", "roimh", "sa", "seacht", "seachtar", "seachtó", "seasca", "seisear", "siad", "sibh", "sinn", "sna", "sé", "sí", "tar", "thar", "thú", "triúr", "trí", "trína", "trínár", "tríocha", "tú", "um", "ár", "é", "éis", "í", "ó", "ón", "óna", "ónár"},
    "id": {"yang", "dan", "di", "dari", "ini", "pada kepada", "ada adalah", "dengan", "untuk", "dalam", "oleh", "sebagai", "juga", "ke", "atau", "tidak", "itu", "sebuah", "tersebut", "dapat", "ia", "telah", "satu", "memiliki", "mereka", "bahwa", "lebih", "karena", "seorang", "akan", "seperti", "secara", "kemudian", "beberapa", "banyak", "antara", "setelah", "yaitu", "hanya", "hingga", "serta", "sama", "dia", "tetapi", "namun", "melalui", "bisa", "sehingga", "ketika", "suatu", "sendiri", "bagi", "semua", "harus", "setiap", "maka", "maupun", "tanpa", "saja", "jika", "bukan", "belum", "sedangkan", "yakni", "meskipun", "hampir", "kita", "demikian", "daripada", "apa", "ialah", "sana", "begitu", "seseorang", "selain", "terlalu", "ataupun", "saya", "bila", "bagaimana", "tapi", "apabila", "kalau", "kami", "melainkan", "boleh", "aku", "anda", "kamu", "beliau", "kalian"},
  }
  if not lang or lang not in stopwords:
    return lambda x: x
  def stopfilter(words: list[str]) -> list[str]:
    ret = []
    if words:
      for word in words:
        w2 = word.lower().strip()
        if w2 not in stopwords[lang]:
          ret.append(w2)

    return ret
  return stopfilter

# default tokenizer: split-on-whitespace + lowercase + remove common punctiation
def tokenizer_default( s ) :
  ltrimchars = ['(','[','{','<','\'','"']
  rtrimchars = ['.', '?', '!', ',', ':', ';', ')', ']', '}', '>','\'','"']
  if type(s) != str : return []
  wl = s.lower().split()
  for i,w in enumerate(wl) :
    if len(w) < 1 : continue
    si = 0
    ei = len(w)
    while si < ei and w[si] in ltrimchars : si += 1
    while ei > si and w[ei-1] in rtrimchars : ei -= 1
    wl[i] = wl[i][si:ei]
  wl = [ w for w in wl if len(w) > 0 ]
  return wl


# simple split-on-whitespace tokenizer, not recommended
def tokenizer_whitespace( s ) :
  return s.split()


# integrated BM25 class with multiple algoritms and options
class BM25opt :
  def __init__( self, corpus, algo='okapi', tokenizer_function=tokenizer_default, stopwords_filter=stop_words_filter(None), idf_algo=None, k1=None, b=None, epsilon=None, delta=None ) :
    # version
    self.version = '1.2.0'
    # tokenizing input
    self.tokenizer_function = lambda x: stopwords_filter(tokenizer_function(x))
    tokenized_corpus = [ self.tokenizer_function( document ) for document in corpus ]
    # algoritm selection
    self.algo = algo.strip().lower()
    if self.algo not in ['okapi','l','plus'] : self.algo = 'okapi'
    self.idf_algo = idf_algo if idf_algo is not None else algo
    if self.idf_algo not in ['okapi','l','plus'] : self.idf_algo = 'okapi'

    # constants
    if algo == 'okapi' :
      self.k1 = k1 if k1 is not None else 1.5
      self.b = b if b is not None else 0.75
      self.epsilon = epsilon if epsilon is not None else 0.25
      self.delta = delta if delta is not None else 1
    if algo == 'l' :
      self.k1 = k1 if k1 is not None else 1.5
      self.b = b if b is not None else 0.75
      self.delta = delta if delta is not None else 0.5
      self.epsilon = epsilon if epsilon is not None else 0.25
    if algo == 'plus' :
      self.k1 = k1 if k1 is not None else 1.5
      self.b = b if b is not None else 0.75
      self.delta = delta if delta is not None else 1
      self.epsilon = epsilon if epsilon is not None else 0.25

    # common
    self.corpus_len = len(corpus)
    self.avg_doc_len = 0
    self.word_freqs = []
    self.doc_lens = []
    self.word_docs_count = {}  # word -> number of documents with word
    self.total_word_count = 0

    for document in tokenized_corpus:
      # doc lengths and total word count
      self.doc_lens.append(len(document))
      self.total_word_count += len(document)
      # word frequencies in this document
      frequencies = {}
      for word in document:
        if word not in frequencies:
          frequencies[word] = 0
        frequencies[word] += 1
      self.word_freqs.append(frequencies)
      # number of documents with word count
      for word, freq in frequencies.items():
        try:
          self.word_docs_count[word] += 1
        except KeyError:
          self.word_docs_count[word] = 1

    # create wsmap
    self.createwsmap()

    ### End of __init__()


  # creating the words * documents score map
  def createwsmap(self) :

    # average document length
    self.avg_doc_len = self.total_word_count / self.corpus_len

    # IDF
    self.idf = {}
    # https://github.com/dorianbrown/rank_bm25/issues/35 : atire IDF correction is possible if idf_algo is set
    if self.idf_algo == 'okapi' :
      idf_sum = 0
      negative_idfs = []
      for word, freq in self.word_docs_count.items():
        idf = math.log(self.corpus_len - freq + 0.5) - math.log(freq + 0.5)
        self.idf[word] = idf
        idf_sum += idf
        if idf < 0:
          negative_idfs.append(word)
      self.average_idf = idf_sum / len(self.idf)
      # assign epsilon
      eps = self.epsilon * self.average_idf
      for word in negative_idfs:
        self.idf[word] = eps
    if self.idf_algo == 'l' : # IDF for BM25L
      for word, doccount in self.word_docs_count.items():
        self.idf[word] = math.log(self.corpus_len + 1) - math.log(doccount + 0.5)
    if self.idf_algo == 'plus' : # IDF for BM25Plus
      for word, doccount in self.word_docs_count.items():
        self.idf[word] = math.log(self.corpus_len + 1) - math.log(doccount)

    # "half divisor"
    self.hds = [ ( 1-self.b + self.b*doc_len/self.avg_doc_len) for doc_len in self.doc_lens ]

    # words * documents score map
    self.wsmap = {}
    for word in self.idf :
      self.wsmap[word] = np.zeros( self.corpus_len )
      for di in range(0,self.corpus_len) :
        twf = (self.word_freqs[di].get(word) or 0)
        if self.algo == 'okapi' :
          self.wsmap[word][di] = self.idf[word] * ( twf * (self.k1 + 1) / ( twf + self.k1 * self.hds[di] ) )
        if self.algo == 'l' :
          self.wsmap[word][di] = self.idf[word] * twf * (self.k1 + 1) * ( twf/self.hds[di] + self.delta) / (self.k1 + twf/self.hds[di] + self.delta)
        if self.algo == 'plus' :
          self.wsmap[word][di] = self.idf[word] * (self.delta + ( twf * (self.k1 + 1) / ( twf + self.k1 * self.hds[di] ) ))

    ### End of createwsmap()


  # get a list of scores for every document
  def get_scores( self, query ) :
    tokenizedquery = self.tokenizer_function( query )
    # zeroes list of scores
    scores = np.zeros( self.corpus_len )
    # for each word in tokenizedquery, if word is in wsmap, lookup and add word score for every documents' scores
    for word in tokenizedquery:
      if word in self.wsmap :
        scores += self.wsmap[word]
    # return scores list (not sorted)
    return scores


  # return [id,score] for the top k documents
  def topk( self, query, k=None ) :
    docscores = self.get_scores( query )
    sisc = [ [i,s] for i,s in enumerate(docscores) ]
    sisc.sort(key=lambda x:x[1],reverse=True)
    if k : sisc = sisc[:k]
    return sisc


  # updating index by adding documents
  def add_documents( self, documents ) :
    new_tokenized_documents = [ self.tokenizer_function( document ) for document in documents ]
    self.corpus_len += len(documents)

    # loop new documents
    for tokenized_document in new_tokenized_documents:
      # doc lengths and total word count
      self.doc_lens.append(len(tokenized_document))
      self.total_word_count += len(tokenized_document)
      # word frequencies in this document
      frequencies = {}
      for word in tokenized_document:
        if word not in frequencies:
          frequencies[word] = 0
        frequencies[word] += 1
      self.word_freqs.append(frequencies)
      # number of documents with word count
      for word, freq in frequencies.items():
        try:
          self.word_docs_count[word] += 1
        except KeyError:
          self.word_docs_count[word] = 1
    
    # create wsmap
    self.createwsmap()

    ### End of add_documents()


  # updating index by deleting documents
  def delete_documents( self, document_ids ) :
    self.corpus_len -= len(document_ids)
    document_ids.sort(reverse=True) # important to delete documents in reverse order

    # loop document-to-delete ids
    for d_id in document_ids :
      if d_id < len(self.doc_lens) :
        # doc lengths and total word count
        self.total_word_count -= self.doc_lens[d_id]
        del self.doc_lens[d_id]
        # word frequencies
        for word in self.word_freqs[d_id] :
          self.word_docs_count[word] -= 1
          # number of documents with word count
          if self.word_docs_count[word] < 1 : del self.word_docs_count[word]
        del self.word_freqs[d_id]
    
    # create wsmap
    self.createwsmap()

    ### End of delete_documents()


  # updating index by changing documents
  def update_documents( self, document_ids, documents ) :
    new_tokenized_documents = [ self.tokenizer_function( document ) for document in documents ]

    # loop document-to-update ids
    for i, u_id in enumerate(document_ids) :
      if i < len(new_tokenized_documents) and u_id < len(self.doc_lens) :
        # doc lengths and total word count
        self.total_word_count -= self.doc_lens[u_id]
        self.doc_lens[u_id] = len(new_tokenized_documents[i])
        self.total_word_count += len(new_tokenized_documents[i])
        # word frequencies : remove old
        for word in self.word_freqs[u_id] :
          self.word_docs_count[word] -= 1
          if self.word_docs_count[word] < 1 : del self.word_docs_count[word]
        # word frequencies : add new
        self.word_freqs[u_id] = {}
        for word in new_tokenized_documents[i]:
          if word not in self.word_freqs[u_id]:
            self.word_freqs[u_id][word] = 0
          self.word_freqs[u_id][word] += 1
        # number of documents with word count
        for word, freq in self.word_freqs[u_id].items():
          try:
            self.word_docs_count[word] += 1
          except KeyError:
            self.word_docs_count[word] = 1

    # create wsmap
    self.createwsmap()

    ### End of update_documents()


  ### End of class BM25opt
