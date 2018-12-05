from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import os
import random
import numpy as np
from tqdm import tqdm
import sys, email
import pandas as pd
import math
import datetime

#########################################################
# Load Enron dataset
#########################################################

ENRON_EMAIL_DATASET_PATH = './emails.csv'

# load enron dataset
import pandas as pd
emails_df = pd.read_csv(ENRON_EMAIL_DATASET_PATH)
print(emails_df.shape)

print('atempt to truncate')
emails_df.truncate(before="A", after="Z", axis="columns")

print(emails_df.shape)

emails_df.head()



#########################################################
# Sort out required email features: date, subject, content
#########################################################

# source https://www.kaggle.com/zichen/explore-enron
## Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

import email
# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails_df['message']))
emails_df.drop('message', axis=1, inplace=True)
# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]
# Parse content from emails
emails_df['Content'] = list(map(get_text_from_email, messages))

# keep only Subject and Content for this exercise
emails_df = emails_df[['Date','Subject','Content']]
#
# #########################################################
# # change wor2vec model to work with Enron emails
# #########################################################
#
#
# point it to our Enron data set
emails_sample_df = emails_df.copy()

import string, re
# clean up subject line
emails_sample_df['Subject'] = emails_sample_df['Subject'].str.lower()
emails_sample_df['Subject'] = emails_sample_df['Subject'].str.replace(r'[^a-z]', ' ')
emails_sample_df['Subject'] = emails_sample_df['Subject'].str.replace(r'\s+', ' ')

# clean up content line
emails_sample_df['Content'] = emails_sample_df['Content'].str.lower()
emails_sample_df['Content'] = emails_sample_df['Content'].str.replace(r'[^a-z]', ' ')
emails_sample_df['Content'] = emails_sample_df['Content'].str.replace(r'\s+', ' ')

# create sentence list
emails_text = (emails_sample_df["Subject"] + ". " + emails_sample_df["Content"]).tolist()

sentences = ' '.join(emails_text)
words = sentences.split()

print('Data size', len(words))


# get unique words and map to glove set
print('Unique word count', len(set(words)))
#
#
# # drop rare words
# vocabulary_size = 5
#
# def build_dataset(words):
#   count = [['UNK', -1]]
#   count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
#   dictionary = dict()
#   for word, _ in count:
#     dictionary[word] = len(dictionary)
#   data = list()
#   unk_count = 0
#   for word in tqdm(words):
#     if word in dictionary:
#       index = dictionary[word]
#     else:
#       index = 0  # dictionary['UNK']
#       unk_count += 1
#     data.append(index)
#   count[0][1] = unk_count
#   reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#   return data, count, dictionary, reverse_dictionary
#
# data, count, dictionary, reverse_dictionary = build_dataset(words)
#
# del words
# print('Most common words (+UNK)', count[:5])
# print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
#
# ####################################################################
# # find matches with glove
# ####################################################################
GLOVE_DATASET_PATH = './glove.840B.300d.txt'
#
# from tqdm import tqdm
# import string
# embeddings_index = {}
# f = open(GLOVE_DATASET_PATH)
# word_counter = 0
# for line in tqdm(f):
#   values = line.split()
#   word = values[0]
#   if word in dictionary:
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
#   word_counter += 1
# f.close()
#
# print('Found %s word vectors matching enron data set.' % len(embeddings_index))
# print('Total words in GloVe data set: %s' % word_counter)#
#
# #########################################################
# # Check out some clusters
# #########################################################
#
# # create a dataframe using the embedded vectors and attach the key word as row header
# import pandas as pd
# enrond_dataframe = pd.DataFrame(embeddings_index)
# enrond_dataframe = pd.DataFrame.transpose(enrond_dataframe)
#
# # See what it learns and look at clusters to pull out major themes in the data
# CLUSTER_SIZE = 3
# # cluster vector and investigate top groups
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=CLUSTER_SIZE)
# cluster_make = kmeans.fit_predict(enrond_dataframe)
#
# labels = kmeans.predict(enrond_dataframe)
# import collections
# cluster_frequency = collections.Counter(labels)
# print(cluster_frequency)
# cluster_frequency.most_common()
#
# clusters = {}
# n = 0
# for item in labels:
#     if item in clusters:
#         clusters[item].append(list(enrond_dataframe.index)[n])
#     else:
#         clusters[item] = [list(enrond_dataframe.index)[n]]
#     n +=1
#
# for k,v in cluster_frequency.most_common(100):
#   print('\n\n')
#   print('Cluster:', k)
#   print (' '.join(clusters[k]))

  ####################################################
# Load master clusters for all six deparatments
####################################################
#
# LEGAL= ['affirmed','alleged','appeal','appealed','appeals','appellate','attorney','attorneys','bankruptcy','case','cases','charged','charges','civil','claim','claims','complaint','constitutional','constitutionality','constitutionally','copyright','counsel','court','courts','criminal','damages','decision','decree','decrees','defendants','denied','dispute','dissented','dissenting','enforcement','federal','filed','filing','invalidate','invalidated','judge','judgement','judges','judgment','judgments','judicial','judiciary','jurisdiction','jurisprudence','justice','justices','law','laws','lawsuit','lawsuits','lawyer','lawyers','legal','legality','legally','litigation','overrule','overruled','overturn','overturned','overturning','plaintiff','precedent','precedents','prosecutorial','reversed','rights','ruled','ruling','rulings','settlement','settlements','sue','supreme','tribunal','tribunals','unanimous','unconstitutional','upheld','uphold','upholding','upholds','verdict','violation']
#
# COMMUICATIONS=['accessed','ads','alphabetical','alphabetically','archive','archived','archives','below','bookmark','bookmarked','bookmarks','browse','browsing','calendar','catalog','categories','categorized','category','chart','charts','check','classified','classifieds','codes','compare','content','database','details','directories','directory','domain','domains','downloadable','entries','favorites','feeds','free','genealogy','homepage','homepages','hosting','index','indexed','indexes','info','information','keyword','keywords','library','link','linking','links','list','listed','listing','listings','lists','locate','locator','maps','online','page','pages','peruse','portal','profile','profiles','rated','related','resource','results','search','searchable','searched','searches','searching','selections','signup','site','sites','sorted','statistics','stats','subscribing','tagged','testimonials','titles','updated','updates','via','web','webmaster','webpage','webpages','website','websites','wishlist','accountant','careers','clerical','contracting','department','employed','employee','employees','employer','employers','employment','experienced','freelance','fulltime','generalist','hire','hired','hires','hiring','hourly','intern','interviewing','job','jobs','labor','labour','managerial','manpower','office','paralegal','personnel','placements','positions','profession','professional','professions','qualified','receptionist','recruit','recruiter','recruiters','recruiting','recruitment','resume','resumes','salaried','salaries','salary','seeking','skilled','staff','staffing','supervisor','trainee','vacancies','vacancy','worker','workers','workforce','workplace']
#
# SECURITY_SPAM_ALERTS= ['abducted','accidental','anthrax','anti','antibiotic','antibiotics','assaulted','attacked','attacker','attackers','auth','authenticated','authentication','avoid','avoidance','avoided','avoiding','bacteria','besieged','biometric','bioterrorism','blocking','boarded','bodyguards','botched','captive','captives','captors','captured','chased','commandeered','compromised','confronted','contagious','cornered','culprit','damage','damaging','danger','dangerous','dangers','destroying','destructive','deterrent','detrimental','disruptive','electrocuted','eliminate','eliminating','encroachment','encrypted','encryption','epidemic','escape','escaped','escapee','escaping','expose','exposed','exposing','fatally','feared','fled','flee','fleeing','flu','foiled','freed','germ','germs','guarded','guarding','guards','gunning','hapless','harassed','harm','harmful','harmless','harsh','hepatitis','hid','hijacked','hijacker','hijackers','hiv','hostage','hostages','hunted','immune','immunity','immunization','imprisoned','improper','inadvertent','infect','infected','infecting','infection','infections','infectious','infects','injuring','intentional','interference','interfering','intruders','intrusion','intrusive','invaded','isolates','kidnapped','limiting','login','logins','logon','lured','malaria','malicious','masked','minimise','minimize','minimizing','misuse','mite','mitigating','mosquito','motorcade','nuisance','offending','outbreak','overrun','passcode','password','passwords','plaintext','pneumonia','policeman','potentially','prevent','prevented','preventing','prevents','prone','protect','protected','protecting','protection','protects','quarantine','raided','ransom','raped','refuge','removing','rescued','rescuing','resisting','risks','robbed','runaway','safeguard','secret','secrets','seized','sensitive','server','shielding','smallpox','spam','spores','stolen','stormed','strain','strains','stranded','strep','summoned','susceptible','swine','threat','threatened','threatening','threats','thwarted','tortured','trapped','unaccounted','undesirable','unhealthy','unidentified','unintended','unintentional','unnamed','unnecessary','unprotected','unsafe','unwanted','unwelcome','user','username','vaccine','vaccines','villagers','viral','virus','viruses','vulnerability','vulnerable','whereabouts','whooping','withstand','wounded']
#
# SUPPORT=['ability','acrobat','adobe','advantage','advice','aid','aids','aim','alternatives','app','apps','ares','assist','autodesk','avs','benefits','best','boost','bring','bringing','build','cad','ccna','cellphone','challenge','choices','choosing','citrix','compatible','computer','computers','conferencing','console','consoles','continue','contribute','corel','create','creating','crucial','desktop','desktops','develop','devices','digital','discover','discuss','ease','easier','educate','effective','effectively','effort','electronic','electronics','encarta','encourage','energy','enhance','ensure','essential','eudora','experience','explore','finding','future','gadget','gadgets','gizmos','goal','groupwise','guide','guides','handhelds','handset','handsets','hardware','help','helpful','helping','helps','hopes','ideas','idm','important','improve','interactive','internet','introduce','intuit','invaluable','ios','join','kiosk','kiosks','laptops','lead','learn','lightwave','mac','machines','macintosh','macromedia','maintain','manage','mcafee','mcse','meet','messaging','metastock','microsoft','mobile','monitors','morpheus','mouse','mice','msie','multimedia','natural','needed','needs','netware','networked','networking','norton','notebooks','novell','ocr','oem','offline','office','opportunity','our','peripherals','personal','pgp','phone','phones','photoshop','plan','plans','portables','potential','practical','prepare','pros','quark','quicken','realplayer','recommend','remotely','resco','resources','safe','save','saving','sbe','screens','serve','servers','share','sharing','software','solve','sophos','spb','spss','ssg','standalone','support','symantec','task','tech','telephones','televisions','their','tips','to','together','trojan','useful','users','valuable','veritas','virtual','visio','vista','vital','vmware','ways','wga','whs','winzip','wordperfect','working','workstation','workstations','xp','xpress']
#
# ENERGY_DESK=['amps','baseload','bhp','biomass','blowers','boiler','boilers','btu','btus','burners','cc','cfm','chiller','chillers','cogen','cogeneration','compressors','conditioner','conditioners','conditioning','coolers','cooling','cranking','desalination','diesels','electric','electrical','electricity','electricty','electrification','energy','engine','engines','furnace','furnaces','gasification','generators','genset','geothermal','gigawatt','gpm','heat','heater','heaters','heating','horsepower','hp','hvac','hydro','hydroelectric','hydroelectricity','hydropower','idle','idling','ignition','interconnectors','intertie','kilovolt','kilowatt','kilowatts','kw','kwh','levelized','liter','megawatt','megawatts','microturbine','microturbines','motor','motors','mph','municipally','peaker','photovoltaic','photovoltaics','power','powered','powerplant','powerplants','psi','psig','reactors','redline','refrigerated','refrigeration','renewable','renewables','repower','repowering','retrofits','retrofitting','revs','rpm','siting','solar','substation','substations','switchgear','switchyard','temperatures','terawatt','thermo','thermoelectric','thermostat','thermostats','throttle','torque','turbine','turbines','turbo','undergrounding','ventilation','volt','volts','weatherization','whp','wind','windmill','windmills','windpower']
#
# SALES_DEPARTMENT=['accounting','actuals','advertised','affordable','auction','auctions','audited','auditing','bargain','bargains','bidding','billable','billed','billing','billings','bookkeeping','bought','brand','branded','brands','broker','brokerage','brokers','budgeting','bulk','buy','buyer','buyers','buying','buys','cancel','cancellation','cancellations','cancelled','cardholders','cashback','cashflow','chain','chargeback','chargebacks','cheap','cheaper','cheapest','checkbook','checkout','cheque','cheques','clearance','closeout','consignment','convenience','cosmetics','coupon','coupons','deals','debit','debited','debits','deducted','delivery','deposit','discontinued','discount','discounted','discounts','distributor','ebay','escrow','expensive','export','exported','exporter','exporters','exporting','exports','fee','fees','goods','gratuities','gratuity','groceries','grocery','import','importation','imported','importer','importers','importing','imports','incur','inexpensive','instore','inventory','invoice','invoiced','invoices','invoicing','item','items','lease','ledger','ledgers','manufacturer','marketed','merchandise','merchant','negotiable','nonmembers','nonrefundable','ordering','origination','outlets','overage','overdraft','overstock','owner','owners','payable','payables','payment','payroll','postage','postmarked','premium','prepaid','prepay','prepayment','price','priced','prices','pricey','pricing','product','products','proforma','purchase','purchased','purchaser','purchases','purchasing','rebate','rebook','rebooked','rebooking','receipts','receivable','receivables','reconciliations','recordkeeping','redeem','redeemable','refund','refundable','refunded','refunding','refunds','remittance','resell','reselling','retail','retailer','retailing','sale','sell','seller','sellers','selling','sells','shipment','shipments','shipped','shipper','shippers','shipping','shop','shopped','shopping','shops','sold','spreadsheets','store','stores','submittals','supermarket','supermarkets','superstore','supplier','supplies','supply','surcharge','surcharges','timesheet','timesheets','transaction','upfront','vending','vendor','verifications','voucher','vouchers','warehouse','warehouses','wholesale','wholesaler','wholesaling']

#LEADING

Networking=['accountants', 'administrators', 'advisers', 'advisors', 'alums', 'analysts', 'appraisers', 'architects', 'arrangers', 'assessors', 'assistants']#, 'attendants', 'auditors', 'backers', 'bankers', 'barbers', 'broadcasters', 'bureaucrats', 'bureaus', 'businessmen', 'businesspeople', 'businesswomen', 'campaigners', 'changers', 'clerks', 'coders', 'collaborators', 'collectors', 'columnists', 'commentators', 'conservationists', 'contributors', 'coordinators', 'correspondents', 'counselors', 'creatives', 'creators', 'dealmakers', 'designers', 'directors', 'dispatchers', 'doers', 'economists', 'educators', 'engineers', 'entertainers', 'entrepreneurs', 'environmentalists', 'examiners', 'execs', 'executives', 'exhibitors', 'facilitators', 'financiers', 'funders', 'generalists', 'geologists', 'greeters', 'gurus', 'handlers', 'helpers', 'historians', 'innovators', 'insiders', 'inspectors', 'integrators', 'intermediaries', 'interns', 'interviewees', 'interviewers', 'inventors', 'invitees', 'janitors', 'journalists', 'keepers', 'landowners', 'librarians', 'lobbyists', 'makers', 'managers', 'mangers', 'marketers', 'mediators', 'mentors', 'meteorologists', 'moderators', 'moguls', 'negotiators', 'newsroom', 'observers', 'operators', 'orgs', 'originators', 'paralegals', 'pickers', 'planners', 'policymakers', 'politicos', 'practitioners', 'principals', 'programmers', 'promoters', 'publicans', 'publics', 'pundits', 'raters', 'realtors', 'recruiters', 'regulators', 'reporters', 'reps', 'reviewers', 'salespeople', 'schedulers', 'screeners', 'secretaries', 'setters', 'signers', 'spokespeople', 'spokespersons', 'staffers', 'strategists', 'subcontractors', 'supervisors', 'surveyed', 'takers', 'tasters', 'techies', 'technologists', 'techs', 'telemarketers', 'testers', 'thinkers', 'treasurers', 'visionaries', 'watchdogs', 'watchers']

Restraining=['abandoning', 'adapting', 'adopting', 'aggregating', 'appreciating', 'assembling', 'auctioning', 'balking', 'beefing', 'bolstering', 'broadening']#, 'brokering', 'bundling', 'capitalizing', 'centralizing', 'championing', 'clarifying', 'commandeering', 'complicating', 'consolidating', 'contemplating', 'curtailing', 'dedicating', 'deliberating', 'devising', 'devoting', 'discounting', 'dismantling', 'distancing', 'diversifying', 'divesting', 'downgrading', 'downsizing', 'duplicating', 'elaborating', 'embarking', 'enlisting', 'envisioning', 'eschewing', 'expediting', 'experimenting', 'exploiting', 'fashioning', 'finalising', 'finalizing', 'flagging', 'fleshing', 'focussing', 'forging', 'forgoing', 'formalizing', 'formulating', 'groundwork', 'gutting', 'harmonization', 'harmonizing', 'harnessing', 'instituting', 'inventing', 'liberalizing', 'maturing', 'merging', 'mobilizing', 'modernizing', 'monetizing', 'mulling', 'navigating', 'optimising', 'orchestrating', 'overhauling', 'perfecting', 'personalizing', 'persuading', 'phasing', 'populating', 'previewing', 'prioritizing', 'procuring', 'rationalizing', 'readying', 'reaffirming', 'reassessing', 'reclaiming', 'reconciling', 'reconsidering', 'recreating', 'rectifying', 'redeeming', 'redefining', 'redrafting', 'reentering', 'reestablishing', 'reforming', 'regaining', 'reinstating', 'reinventing', 'relinquishing', 'renaming', 'renegotiating', 'renewing', 'reorganizing', 'repositioning', 'reserving', 'reshaping', 'restating', 'rethinking', 'retooling', 'revamping', 'reverting', 'revising', 'revisiting', 'revitalizing', 'reviving', 'reworking', 'rewriting', 'scrapping', 'scrutinizing', 'segmenting', 'shoring', 'shying', 'simplifying', 'solidifying', 'standardizing', 'streamlining', 'stressing', 'supplementing', 'tinkering', 'transfering', 'transitioning', 'tweaking', 'uncovering', 'underscoring', 'uniting', 'validating', 'valuing', 'visualizing', 'zeroing']

Facilitating=['able', 'accept', 'add', 'advantage', 'advice', 'advise', 'afford', 'all', 'allow', 'allowed', 'allowing', 'any', 'anytime', 'anywhere', 'are']#, 'arrange', 'ask', 'attempt', 'attempting', 'avoid', 'aware', 'be', 'become', 'begin', 'best', 'better', 'bring', 'build', 'call', 'calling', 'can', 'cannot', 'capture', 'careful', 'carry', 'chance', 'chances', 'change', 'changing', 'check', 'choice', 'choices', 'choose', 'choosing', 'clear', 'collect', 'come', 'compare', 'consider', 'continue', 'correct', 'create', 'deal', 'decide', 'deciding', 'different', 'difficult', 'discover', 'discuss', 'do', 'easier', 'easiest', 'easily', 'easy', 'either', 'enter', 'everyday', 'exact', 'expect', 'express', 'extra', 'fail', 'fair', 'fill', 'find', 'finding', 'follow', 'gather', 'get', 'give', 'giving', 'go', 'have', 'help', 'here', 'hold', 'if', 'important', 'instead', 'interested', 'introduce', 'job', 'keep', 'keeping', 'learn', 'leave', 'let', 'lets', 'likely', 'live', 'll', 'locate', 'longer', 'look', 'looking', 'lose', 'make', 'making', 'manage', 'many', 'may', 'meet', 'money', 'more', 'most', 'must', 'necessary', 'need', 'needed', 'needing', 'needs', 'no', 'not', 'notice', 'ones', 'opt', 'option', 'options', 'or', 'order', 'other', 'others', 'our', 'ourselves', 'own', 'people', 'perform', 'person', 'personal', 'pick', 'place', 'places', 'please', 'possible', 'prefer', 'prepare', 'promise', 'proper', 'put', 'quickly', 'ready', 'real', 'reasons', 'recognize', 'recommend', 'regularly', 'rely', 'repeat', 'replace', 'require', 'respond', 'rid', 'satisfied', 'save', 'saving', 'see', 'seek', 'serve', 'settle', 'share', 'should', 'simple', 'simply', 'solve', 'some', 'speak', 'spend', 'start', 'stay', 'suggest', 'take', 'taking', 'tend', 'their', 'them', 'themselves', 'these', 'they', 'those', 'tips', 'to', 'together', 'trust', 'try', 'unable', 'unless', 'us', 'use', 'using', 'visit', 'want', 'way', 'ways', 'we', 'well', 'whenever', 'wherever', 'whether', 'will', 'willing', 'wish', 'without', 'work', 'you', 'your', 'yourself']

Motivating=['abilities', 'accomplish', 'accomplished', 'accomplishing', 'accomplishments', 'achievement', 'achievements', 'actively', 'addressing', 'advancement']#, 'advancing', 'aim', 'aimed', 'aiming', 'aims', 'ambitions', 'approach', 'approaches', 'aspects', 'aspirations', 'attention', 'attract', 'attracting', 'avenues', 'awareness', 'becoming', 'beyond', 'blueprint', 'branding', 'bringing', 'broader', 'career', 'careers', 'challenge', 'challenges', 'challenging', 'commitment', 'committed', 'communicate', 'communicating', 'competitive', 'conducive', 'continuing', 'contribute', 'contributing', 'creating', 'creation', 'creative', 'creativity', 'critical', 'crucial', 'cultivating', 'demands', 'demonstrate', 'develop', 'developed', 'developing', 'development', 'devoted', 'discovering', 'diverse', 'diversity', 'driven', 'educate', 'educating', 'effort', 'embracing', 'emphasis', 'empowering', 'empowerment', 'encourage', 'encouraged', 'encouragement', 'encourages', 'encouraging', 'endeavor', 'endeavors', 'engage', 'engaged', 'engagement', 'engaging', 'enriching', 'entrepreneurial', 'evolving', 'expanding', 'expectations', 'experience', 'experiences', 'exploration', 'explore', 'exploring', 'focus', 'focused', 'focuses', 'focusing', 'focussed', 'formative', 'foster', 'fostered', 'fostering', 'fosters', 'friendships', 'fruitful', 'fulfill', 'fulfilling', 'furthering', 'future', 'gaining', 'geared', 'generation', 'goal', 'goals', 'growing', 'guiding', 'habits', 'helping', 'horizons', 'ideas', 'importance', 'importantly', 'increasingly', 'influence', 'influencing', 'innovative', 'insight', 'insights', 'inspire', 'interests', 'introducing', 'invaluable', 'involved', 'issues', 'knowledge', 'leadership', 'lifelong', 'lifestyle', 'lifestyles', 'meaningful', 'mentoring', 'milestones', 'mindset', 'motivate', 'motivated', 'motivating', 'motivation', 'niche', 'nontraditional', 'nurturing', 'opportunities', 'opportunity', 'parenting', 'peers', 'perspective', 'positive', 'possibilities', 'potential', 'practical', 'practices', 'productive', 'progress', 'progressing', 'promote', 'promoting', 'prospects', 'pursue', 'pursuing', 'pursuit', 'pursuits', 'raising', 'recognition', 'recognizing', 'relationship', 'relationships', 'researching', 'resources', 'reward', 'rewarding', 'rewards', 'role', 'roles', 'seeking', 'shaping', 'shared', 'sharing', 'social', 'solving', 'stimulating', 'strategies', 'strategy', 'strengths', 'strive', 'strives', 'striving', 'strong', 'succeed', 'success', 'successes', 'successful', 'tackling', 'talent', 'talents', 'tangible', 'targeted', 'targets', 'thrive', 'toward', 'towards', 'transformation', 'transforming', 'ultimately', 'understanding', 'valuable', 'viable', 'vision', 'vital', 'wealth', 'workplace', 'worthwhile', 'youth']

PhysicalAction=['across', 'ahead', 'aisle', 'along', 'apart', 'approaching', 'arms', 'around', 'away', 'back', 'backing', 'backs', 'backward', 'backwards', 'ball']#, 'balls', 'behind', 'bell', 'bend', 'big', 'bite', 'blast', 'block', 'blocks', 'bottom', 'bounce', 'break', 'breaking', 'broken', 'bump', 'bust', 'catch', 'catching', 'caught', 'chunk', 'circle', 'circles', 'close', 'closer', 'closest', 'corner', 'corners', 'counter', 'crawl', 'cross', 'cue', 'cut', 'cuts', 'deep', 'dig', 'direction', 'directions', 'ditch', 'down', 'drag', 'draw', 'drive', 'drop', 'dropping', 'dump', 'edge', 'empty', 'ends', 'face', 'faces', 'facing', 'fall', 'falling', 'farther', 'fast', 'finish', 'flag', 'flip', 'foot', 'fore', 'forward', 'front', 'gap', 'grab', 'ground', 'half', 'halfway', 'hand', 'hands', 'hang', 'hanging', 'haul', 'head', 'heading', 'heads', 'heavy', 'hide', 'hit', 'holding', 'hole', 'holes', 'hooked', 'huge', 'inside', 'into', 'jog', 'jump', 'keeper', 'kick', 'knock', 'lap', 'lay', 'lean', 'leap', 'leaving', 'left', 'line', 'lines', 'long', 'loose', 'mid', 'middle', 'midway', 'minute', 'move', 'moving', 'narrow', 'notch', 'off', 'onto', 'open', 'opposite', 'out', 'outside', 'pace', 'pass', 'passing', 'path', 'pause', 'pinch', 'pit', 'pits', 'point', 'pointing', 'pose', 'pull', 'punch', 'push', 'quick', 'reach', 'reaching', 'rest', 'resting', 'reverse', 'right', 'roll', 'rolling', 'rough', 'round', 'row', 'run', 'running', 'rush', 'scramble', 'seconds', 'shake', 'sharp', 'shot', 'shots', 'shut', 'side', 'sides', 'sight', 'sit', 'sitting', 'skip', 'slack', 'slide', 'slides', 'slightly', 'slip', 'slow', 'slowly', 'snag', 'spin', 'split', 'spot', 'spots', 'spread', 'squeeze', 'stall', 'stand', 'standing', 'steady', 'step', 'stepping', 'steps', 'stick', 'stop', 'stopping', 'straight', 'stride', 'stuck', 'sweep', 'swing', 'tap', 'then', 'throw', 'thru', 'tick', 'tied', 'till', 'tiny', 'tip', 'top', 'toss', 'touch', 'tough', 'trailing', 'trap', 'triangle', 'trick', 'turn', 'turning', 'underneath', 'up', 'upside', 'walk', 'walking', 'whilst', 'zone', 'bare', 'baring', 'bellies', 'bending', 'bent', 'biting', 'blasting', 'blowing', 'bouncing', 'bowing', 'bracing', 'bucking', 'bumping', 'burly', 'busily', 'busting']#, 'carving', 'chew', 'choke', 'choking', 'chomping', 'chopping', 'circling', 'claw', 'claws', 'clinging', 'clipping', 'cracking', 'cranking', 'crawling', 'crouching', 'crunching', 'crushing', 'cutting', 'digging', 'dragging', 'erect', 'eyeballs', 'finger', 'fingers', 'fist', 'flashing', 'flipping', 'fours', 'furiously', 'gash', 'grabbing', 'grappling', 'grind', 'grinding', 'habit', 'hammer', 'hammering', 'hammers', 'handing', 'hauling', 'hilt', 'hooking', 'hopping', 'hugging', 'hump', 'hustling', 'jab', 'jabs', 'jacking', 'juggling', 'jumping', 'kicking', 'knocking', 'laying', 'leaning', 'leaping', 'lifting', 'limp', 'looping', 'milking', 'mouths', 'nailing', 'necks', 'nip', 'noses', 'parting', 'picking', 'piercing', 'piling', 'pinching', 'pointy', 'poke', 'poking', 'popping', 'pounding', 'pressing', 'probing', 'prodding', 'pry', 'pulling', 'punching', 'pushing', 'racking', 'raking', 'reins', 'ripping', 'rocking', 'rooting', 'ropes', 'rounding', 'rubbing', 'sagging', 'scrambling', 'scraping', 'scratching', 'shaking', 'sharpened', 'shovel', 'shovels', 'shoving', 'shredding', 'shreds', 'sideways', 'skipping', 'slamming', 'slapping', 'slashing', 'slippery', 'slipping', 'smashing', 'snapping', 'spinning', 'splitting', 'spotting', 'squat', 'squeezing', 'stab', 'sticking', 'stiff', 'straddling', 'straining', 'stretching', 'strokes', 'stump', 'swaying', 'swinging', 'swipe', 'taping', 'tapping', 'tearing', 'throwing', 'thrust', 'tipping', 'tongues', 'tossing', 'touching', 'trimming', 'tripping', 'twisting', 'tying', 'urinating', 'wagging', 'wheeling', 'whip', 'whipping', 'wiggle', 'wiping', 'wobbly', 'wringing', 'yank','aces', 'against', 'battled', 'beat', 'beating', 'berth', 'birdie', 'bogey', 'bogeys', 'breakaway', 'champ', 'champion', 'champions', 'championship', 'championships', 'champs', 'clash', 'clinch', 'clinching', 'comeback', 'compete', 'competed', 'competes', 'competing', 'competition', 'competitor', 'competitors', 'consecutive', 'consolation', 'contender', 'contenders', 'contest', 'contestant', 'contestants', 'contested', 'contests', 'decisive', 'defeat', 'defeated', 'defeating', 'defeats', 'defending', 'derby', 'discus', 'division', 'divisional', 'dominated', 'dominating', 'doubles', 'earned', 'eighth', 'eventual', 'fielded', 'fifth', 'final', 'finalist', 'finalists', 'finals', 'finishes', 'finishing', 'fourth', 'handicap', 'handily', 'heavyweight', 'hurdle', 'invitational', 'javelin', 'lone', 'longshot', 'losing', 'match', 'matches', 'matchup', 'matchups', 'medal', 'medals', 'narrowly', 'ninth', 'opener', 'opponent', 'opponents', 'overcame', 'pairings', 'pennant', 'penultimate', 'pitted', 'playoff', 'podium', 'postseason', 'putt', 'qualifier', 'qualifiers', 'qualifying', 'race', 'races', 'ranked', 'ranking', 'rankings', 'reigning', 'rival', 'rivalry', 'rivals', 'rout', 'runner', 'runners', 'score', 'scores', 'seeded', 'semifinals', 'seventh', 'shootout', 'showdown', 'singles', 'sixth', 'slam', 'sprint', 'sprinters', 'standings', 'streak', 'tally', 'tiebreaker', 'title', 'toughest', 'tournament', 'tourney', 'triumph', 'trophy', 'unbeaten', 'undefeated', 'underdog', 'upsets', 'victories', 'victory', 'vs', 'win', 'winner', 'winners', 'winning', 'wins', 'won','ably', 'abundantly', 'acutely', 'adequately', 'aggressively', 'alternately', 'amicably', 'amply', 'appropriately', 'arbitrarily', 'artificially', 'attentively', 'attractively']#, 'bilaterally', 'boldly', 'broadly', 'carefully', 'cautiously', 'channeled', 'cheaply', 'cleanly', 'cleverly', 'cohesive', 'comfortably', 'competitively', 'comprehensively', 'conceptually', 'concurrently', 'confidently', 'consciously', 'conservatively', 'consistently', 'conspicuously', 'constantly', 'constructively', 'continually', 'continuously', 'cooperatively', 'correspondingly', 'creatively', 'critically', 'decisively', 'deeply', 'diligently', 'distinctly', 'endlessly', 'energized', 'enormously', 'enthusiastically', 'evenly', 'excessively', 'expeditiously', 'extensively', 'faithfully', 'favorably', 'feverishly', 'fiercely', 'firmly', 'fleshed', 'forcefully', 'freely', 'gradually', 'handsomely', 'heavily', 'honed', 'hotly', 'immersed', 'incrementally', 'independently', 'ineffectively', 'inexpensively', 'informally', 'infrequently', 'intelligently', 'intensely', 'interestingly', 'intimately', 'intuitively', 'keenly', 'labored', 'legitimately', 'liberally', 'loosely', 'massively', 'matured', 'mightily', 'moderately', 'modestly', 'nimbly', 'optimally', 'ostensibly', 'passively', 'persistently', 'piecemeal', 'positively', 'predictably', 'proactively', 'productively', 'profitably', 'progressively', 'prudently', 'purposefully', 'radically', 'rapidly', 'rationally', 'realistically', 'relentlessly', 'reliably', 'remarkably', 'respectfully', 'responsibly', 'restrained', 'rigorously', 'routinely', 'satisfactorily', 'scrutinized', 'sensibly', 'simultaneously', 'smoothly', 'solidly', 'sparingly', 'squarely', 'strategically', 'subtly', 'sufficiently', 'suitably', 'swiftly', 'systematically', 'tended', 'thoroughly', 'thoughtfully', 'tightly', 'tirelessly', 'transparently', 'tremendously', 'uniformly', 'vastly', 'vigorously', 'vitally', 'weakly', 'wisely']

Praise=['acclaim', 'accolades', 'admired', 'aided', 'alongside', 'amalgamation', 'amassed', 'amassing', 'amongst', 'arguably', 'astounding', 'attracted', 'behemoth', 'bevy']#, 'biggest', 'boast', 'boasted', 'boasting', 'boldest', 'bonafide', 'boon', 'breakthrough', 'brightest', 'broadest', 'budding', 'burgeoning', 'clout', 'colossal', 'cornerstone', 'counterpart', 'counterparts', 'coveted', 'culmination', 'cyberspace', 'decade', 'destined', 'distinguished', 'dominance', 'dominate', 'eighties', 'eleventh', 'elite', 'elusive', 'emerge', 'endowed', 'enormous', 'envisioned', 'erstwhile', 'esteemed', 'fabled', 'fame', 'famed', 'famous', 'favored', 'favoured', 'flagship', 'fledged', 'fledgling', 'flourish', 'foothold', 'foray', 'forays', 'forefront', 'foremost', 'formidable', 'fortunes', 'franchises', 'fruition', 'gargantuan', 'garner', 'garnered', 'garnering', 'geniuses', 'giants', 'globe', 'greatest', 'groundbreaking', 'hailed', 'heavyweights', 'helm', 'heralded', 'homage', 'homegrown', 'hugely', 'illustrious', 'imaginable', 'immense', 'inception', 'indispensable', 'influential', 'inroads', 'irreplaceable', 'landmark', 'launching', 'legacy', 'legendary', 'limelight', 'lofty', 'longstanding', 'lucrative', 'luminaries', 'mammoth', 'marketable', 'marvel', 'mecca', 'milestone', 'millionth', 'minted', 'moniker', 'monumental', 'myriad', 'newcomer', 'newcomers', 'newest', 'newfound', 'nineties', 'notable', 'notably', 'noteworthy', 'notoriety', 'overlooked', 'peerless', 'perfected', 'personalities', 'phenomenal', 'pinnacle', 'pioneered', 'pioneering', 'pioneers', 'plethora', 'poised', 'popular', 'popularity', 'powerhouse', 'powerhouses', 'predecessor', 'predecessors', 'preeminent', 'prestige', 'prime', 'prized', 'prolific', 'prominence', 'prominent', 'promising', 'prospect', 'prowess', 'ranks', 'reckoned', 'renowned', 'reputation', 'reputations', 'revolutionized', 'rewarded', 'richest', 'rumored', 'shortlist', 'sizable', 'sizeable', 'slew', 'spearhead', 'spotlight', 'springboard', 'stateside', 'stature', 'stellar', 'stints', 'strides', 'strongest', 'succeeding', 'superpower', 'superstars', 'surpass', 'surpasses', 'surpassing', 'synonymous', 'teamed', 'teaming', 'testament', 'thrived', 'thriving', 'touted', 'touting', 'tremendous', 'ubiquitous', 'unbeatable', 'undiscovered', 'undisputed', 'undoubtedly', 'unequaled', 'unheard', 'unofficial', 'unparalleled', 'unprecedented', 'unrivaled', 'unsung', 'untapped', 'unveil', 'unveiled', 'unveiling', 'upstart', 'vanguard', 'vast', 'vaunted', 'venerable', 'vying', 'workhorse', 'worthy']

Criticism=['abnormally', 'abrupt', 'absence', 'accelerated', 'accidental', 'accumulate', 'accumulation', 'adverse', 'adversely', 'affect', 'affected', 'affecting', 'affects']#, 'aging', 'alteration', 'alterations', 'apparent', 'associated', 'attributable', 'attributed', 'attrition', 'avoidance', 'avoided', 'behavior', 'behaviour', 'bodily', 'breakage', 'breakdown', 'cause', 'caused', 'causes', 'causing', 'cessation', 'changes', 'circulation', 'compensate', 'compensatory', 'concurrent', 'condition', 'conditions', 'consequence', 'consequences', 'consequent', 'continual', 'contraction', 'damage', 'decay', 'decrease', 'decreased', 'decreases', 'decreasing', 'defect', 'defective', 'defects', 'deficits', 'dependence', 'depletion', 'deterioration', 'difficulty', 'diminished', 'disrupted', 'disruption', 'disturbance', 'diversion', 'drastic', 'due', 'effect', 'effected', 'effects', 'elevated', 'elimination', 'erosion', 'exacerbated', 'excessive', 'exhaustion', 'experiencing', 'exposure', 'exposures', 'extreme', 'factors', 'failure', 'fluctuating', 'fluctuations', 'frequent', 'gradual', 'heightened', 'imbalance', 'immediate', 'impacted', 'impaired', 'impairment', 'impairments', 'inability', 'inactivity', 'indication', 'indications', 'indicative', 'indirect', 'infancy', 'inflow', 'infrequent', 'instability', 'insufficient', 'intermittent', 'interruption', 'involuntary', 'irregular', 'lack', 'likelihood', 'loss', 'malfunction', 'minor', 'moderate', 'mortality', 'negative', 'negatively', 'neglect', 'negligible', 'normal', 'noticeable', 'obstruction', 'occur', 'occuring', 'occurrence', 'occurrences', 'occurring', 'occurs', 'outflow', 'overload', 'overuse', 'owing', 'partial', 'periods', 'persist', 'persistent', 'persists', 'physical', 'potentially', 'premature', 'pressures', 'problems', 'prolonged', 'prone', 'propensity', 'rapid', 'recurring', 'reduced', 'reduction', 'relief', 'repeated', 'result', 'resulted', 'resulting', 'reversal', 'risk', 'risks', 'secondary', 'severely', 'severity', 'shrinkage', 'significant', 'slight', 'sporadic', 'stagnation', 'stress', 'stresses', 'subsequent', 'sudden', 'susceptible', 'sustained', 'systemic', 'temporary', 'tendency', 'tolerated', 'transient', 'trigger', 'triggered', 'triggering', 'triggers', 'unaffected', 'uncontrollable', 'uncontrolled', 'undetermined', 'unexplained', 'unfavorable', 'unspecified', 'weakness', 'widespread', 'withdrawal', 'worsen', 'worsening','acrimonious', 'afloat', 'ailing', 'ails', 'awry', 'backsliding', 'beleaguered', 'blunder', 'bogged', 'brink', 'buffeted', 'bungled', 'bungling', 'chalked', 'clouded', 'crippled', 'crumbling', 'dangerously', 'deadlock', 'decimated', 'derailed', 'desparately', 'deteriorating', 'dicey', 'disarray', 'dogged', 'doldrums', 'doomed', 'eke', 'embroiled', 'extricate', 'falter', 'faltered', 'faltering', 'fated', 'floundering', 'footing', 'foundering', 'fragile', 'fray', 'fumbling', 'gambit', 'gridlock', 'haggling', 'hampered', 'handedly', 'hardball', 'haywire', 'headway', 'hemorrhaging', 'hindered', 'hobbled', 'hopelessly', 'imbroglio', 'impasse', 'impenetrable', 'implode', 'imploded', 'irreparably', 'jeopardized', 'jeopardy', 'jockeying', 'languish', 'languished', 'languishing', 'lifeline', 'limbo', 'limping', 'linchpin', 'littered', 'logjam', 'loomed', 'maneuverings', 'marred', 'materialise', 'mired', 'morass', 'moribund', 'muddle', 'nary', 'overburdened', 'overheated', 'panacea', 'paralyzed', 'pariah', 'perilously', 'politicking', 'precarious', 'precipice', 'prematurely', 'quagmire', 'reeling', 'riddled', 'roadblock', 'roiled', 'rut', 'saddled', 'scuttle', 'scuttled', 'shambles', 'sidelined', 'snafu', 'soured', 'souring', 'squabbling', 'stalemate', 'stalled', 'stalling', 'stave', 'stifled', 'stifling', 'straits', 'stubbornly', 'stumbling', 'stymied', 'suitors', 'swamped', 'tailspin', 'tangle', 'tangled', 'tarnished', 'tatters', 'teetering', 'tenuous', 'thorny', 'thwarted', 'tightrope', 'topple', 'torrid', 'tragically', 'treading', 'tussle', 'uncharted', 'undoing', 'undone', 'unfulfilled', 'unnoticed', 'unravel', 'unraveled', 'unraveling', 'unscathed', 'verge', 'wayside', 'wither', 'withering', 'withstood', 'wrangle', 'wrangles', 'wrangling']






# boost bags with cosine distance from full glove data set
from tqdm import tqdm
import string
embeddings_index = {}
f = open(GLOVE_DATASET_PATH)
word_counter = 0
for line in tqdm(f):
  values = line.split()
  word = values[0]
  # difference here as we don't intersect words, we take most of them
  if (word.islower() and word.isalpha()): # work with smaller list of vectors
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
  word_counter += 1
f.close()

print('Found %s word vectors matching enron data set.' % len(embeddings_index))
print('Total words in GloVe data set: %s' % word_counter)

# create a dataframe using the embedded vectors and attach the key word as row header
glove_dataframe = pd.DataFrame(embeddings_index)
glove_dataframe = pd.DataFrame.transpose(glove_dataframe)

#departments = [LEGAL, COMMUICATIONS, SECURITY_SPAM_ALERTS, SUPPORT, ENERGY_DESK, SALES_DEPARTMENT]

departments = [Networking, Restraining, Facilitating, Motivating, PhysicalAction, Praise, Criticism]

temp_matrix = pd.DataFrame.as_matrix(glove_dataframe)
import scipy
import scipy.spatial

vocab_boost_count = 5
for group_id in range(len(departments)):
  print('Working bag number:', str(group_id))
  glove_dataframe_temp = glove_dataframe.copy()
  vocab = []
  for word in departments[group_id]:
    print(word)
    vocab.append(word)
    cos_dist_rez = scipy.spatial.distance.cdist(temp_matrix, np.array(glove_dataframe.loc[word])[np.newaxis,:], metric='cosine')
    # find closest words to help
    glove_dataframe_temp['cdist'] = cos_dist_rez
    glove_dataframe_temp = glove_dataframe_temp.sort_values(['cdist'], ascending=[1])
    vocab = vocab + list(glove_dataframe_temp.head(vocab_boost_count).index)
  # replace boosted set to old department group and remove duplicates
  departments[group_id] = list(set(vocab))

# save final objects to disk
import cPickle as pickle
with open('full_bags.pk', 'wb') as handle:
  pickle.dump(departments, handle)




#####################################################################
# Create features of word counts for each department in each email
#####################################################################

import cPickle as pickle
with open('full_bags.pk', 'rb') as handle:
    departments = pickle.load(handle)

# loop through all emails and count group words in each raw text
words_groups = []
for group_id in range(len(departments)):
  work_group = []
  print('Working bag number:', str(group_id))
  top_words = departments[group_id]
  for index, row in tqdm(emails_sample_df.iterrows()):
    text = (row["Subject"] + " " + row["Content"])
    work_group.append(len(set(top_words) & set(text.split())))
    #work_group.append(len([w for w in text.split() if w in set(top_words)]))

  words_groups.append(work_group)

# count emails per category group and feature engineering

raw_text = []
subject_length = []
subject_word_count = []
content_length = []
content_word_count = []
is_am_list = []
is_weekday_list = []
group_Networking = []
group_Restraining = []
group_Facilitating = []
group_Motivating = []
group_PhysicalAction = []
group_Praise = []
group_Criticism = []
final_outcome = []


emails_sample_df['Subject'].fillna('', inplace=True)
emails_sample_df['Date'] = pd.to_datetime(emails_sample_df['Date'], infer_datetime_format=True)

counter = 0
for index, row in tqdm(emails_sample_df.iterrows()):
  raw_text.append([row["Subject"] + " " + row["Content"]])
  group_Networking.append(words_groups[0][counter])
  group_Restraining.append(words_groups[1][counter])
  group_Facilitating.append(words_groups[2][counter])
  group_Motivating.append(words_groups[3][counter])
  group_PhysicalAction.append(words_groups[4][counter])
  group_Praise.append(words_groups[5][counter])
  group_Criticism.append(words_groups[6][counter])
  outcome_tots = [words_groups[0][counter], words_groups[1][counter], words_groups[2][counter],
    words_groups[3][counter], words_groups[4][counter], words_groups[5][counter], words_groups[6][counter]]
  final_outcome.append(outcome_tots.index(max(outcome_tots)))

  subject_length.append(len(row['Subject']))
  subject_word_count.append(len(row['Subject'].split()))
  content_length.append(len(row['Content']))
  content_word_count.append(len(row['Content'].split()))
  dt = row['Date']
  is_am = 'no'
  if (dt.time() < datetime.time(12)): is_am = 'yes'
  is_am_list.append(is_am)
  is_weekday = 'no'
  if (dt.weekday() < 6): is_weekday = 'yes'
  is_weekday_list.append(is_weekday)
  counter += 1


# add simple engineered features?
training_set = pd.DataFrame({
              "raw_text":raw_text,
              "group_Networking":group_Networking,
              "group_Restraining":group_Restraining,
              "group_Facilitating":group_Facilitating,
              "group_Motivating":group_Motivating,
              "group_PhysicalAction":group_PhysicalAction,
              "group_Praise":group_Praise,
              "group_Criticism":group_Criticism,
              "subject_length":subject_length,
              "subject_word_count":subject_word_count,
              "content_length":content_length,
              "content_word_count":content_word_count,
              "is_AM":is_am_list,
              "is_weekday":is_weekday_list,
              "outcome":final_outcome})

# remove all emails that have all zeros (i.e. not from any of required categories)
training_set = training_set[(training_set.group_Networking > 0) |
              (training_set.group_Restraining > 0) |
              (training_set.group_Facilitating > 0) |
              (training_set.group_Motivating > 0) |
              (training_set.group_PhysicalAction > 0) |
              (training_set.group_Praise > 0) |
              (training_set.group_Criticism > 0)]
print(len(training_set))

# save extractions to file
training_set.to_csv('enron_classification_df.csv', index=False, header=True)


####################################################
# TensorFlow Deep Classifier
####################################################
# https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier
# create a wide and deep model and also predict a few entries

model_ready_data = pd.read_csv('enron_classification_df.csv')

# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
# (60% - train set, 20% - validation set, 20% - test set)
df_train, df_test, df_val = np.split(model_ready_data.sample(frac=1), [int(.6*len(model_ready_data)), int(.8*len(model_ready_data))])

# Continuous base columns
content_length = tf.contrib.layers.real_valued_column("content_length")
content_word_count = tf.contrib.layers.real_valued_column("content_word_count")
subject_length = tf.contrib.layers.real_valued_column("subject_length")
subject_word_count = tf.contrib.layers.real_valued_column("subject_word_count")
group_Networking = tf.contrib.layers.real_valued_column("group_Networking")
group_Restraining = tf.contrib.layers.real_valued_column("group_Restraining")
group_Facilitating = tf.contrib.layers.real_valued_column("group_Facilitating")
group_Motivating = tf.contrib.layers.real_valued_column("group_Motivating")
group_PhysicalAction = tf.contrib.layers.real_valued_column("group_PhysicalAction")
group_Praise = tf.contrib.layers.real_valued_column("group_Praise")
group_Criticism = tf.contrib.layers.real_valued_column("group_Criticism")
content_length_bucket = tf.contrib.layers.bucketized_column(content_length, boundaries=[100, 200, 300, 400])
subject_length_bucket = tf.contrib.layers.bucketized_column(subject_length, boundaries=[10,15, 20, 25, 30])

# Categorical base columns
is_AM_sparse_column = tf.contrib.layers.sparse_column_with_keys(column_name="is_AM", keys=["yes", "no"])
# is_AM = tf.contrib.layers.one_hot_column(is_AM_sparse_column)\
is_weekday_sparse_column = tf.contrib.layers.sparse_column_with_keys(column_name="is_weekday", keys=["yes", "no"])
# is_weekday = tf.contrib.layers.one_hot_column(is_weekday_sparse_column)

categorical_columns = [is_AM_sparse_column, is_weekday_sparse_column, content_length_bucket, subject_length_bucket]

deep_columns = [content_length, content_word_count, subject_length, subject_word_count,
               group_Networking, group_Restraining, group_Facilitating, group_Motivating,
               group_PhysicalAction, group_Praise, group_Criticism]

simple_columns = [group_Networking, group_Restraining, group_Facilitating, group_Motivating,
                group_PhysicalAction, group_Praise, group_Criticism]

import tempfile
model_dir = tempfile.mkdtemp()
classifier = tf.contrib.learn.DNNClassifier(feature_columns=simple_columns,
                                hidden_units=[20, 20],
                                n_classes=7,
                                model_dir=model_dir,)

# Define the column names for the data sets.
COLUMNS = ['content_length',
 'content_word_count',
 'group_Networking',
 'group_Restraining',
 'group_Facilitating',
 'group_Motivating',
 'group_PhysicalAction',
 'group_Praise',
 'group_Criticism',
 'is_AM',
 'is_weekday',
 'subject_length',
 'subject_word_count',
 'outcome']
LABEL_COLUMN = 'outcome'
CATEGORICAL_COLUMNS = ["is_AM", "is_weekday"]
CONTINUOUS_COLUMNS = ['content_length',
 'content_word_count',
 'group_Networking',
 'group_Restraining',
 'group_Facilitating',
 'group_Motivating',
 'group_PhysicalAction',
 'group_Praise',
 'group_Criticism',
 'subject_length',
 'subject_word_count']

LABELS = [0, 1, 2, 3, 4, 5, 6]

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())

  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)
# After reading in the data, you can train and evaluate the model:

classifier.fit(input_fn=train_input_fn, steps=200)
results = classifier.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

y_pred = classifier.predict(input_fn=lambda: input_fn(df_val), as_iterable=False)
print(y_pred)

print('buckets found:',set(y_pred))

# # confusion matrix analysis
# from sklearn.metrics import confusion_matrix
# confusion_matrix(df_val[LABEL_COLUMN], y_pred)
# pd.crosstab(df_val[LABEL_COLUMN], y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

# create some data
# https://stats.stackexchange.com/questions/95209/how-can-i-interpret-sklearn-confusion-matrix
lookup = {0: 'Networking', 1:'Restraining', 2:'Facilitating', 3:'Motivating', 4:'PhysicalAction', 5:'Praise', 6:'Criticism'}
y_truet = pd.Series([lookup[_] for _ in df_val[LABEL_COLUMN]])
y_predt = pd.Series([lookup[_] for _ in y_pred])
pd.crosstab(y_truet, y_predt, rownames=['Actual'], colnames=['Predicted'], margins=True)

subject_to_predict = "To the help desk"
content_to_predict = "My monitor stopped responding and I need to get this spreadsheet finished as soon as possible. Please help me!"

def scrub_text(subject_to_predict, content_to_predict, departments):
  # prep text
  subject_to_predict = subject_to_predict.lower()
  pattern = re.compile('[^a-z]')
  subject_to_predict = re.sub(pattern, ' ', subject_to_predict)
  pattern = re.compile('\s+')
  subject_to_predict = re.sub(pattern, ' ', subject_to_predict)

  content_to_predict = content_to_predict.lower()
  pattern = re.compile('[^a-z]')
  content_to_predict = re.sub(pattern, ' ', content_to_predict)
  pattern = re.compile('\s+')
  content_to_predict = re.sub(pattern, ' ', content_to_predict)

  # get bag-of-words
  words_groups = []
  text = subject_to_predict + ' ' + content_to_predict
  for group_id in range(len(departments)):
    work_group = []
    print('Working bag number:', str(group_id))
    top_words = departments[group_id]
    work_group.append(len([w for w in text.split() if w in set(top_words)]))
    words_groups.append(work_group)

  # count emails per category group and feature engineering
  raw_text = []
  subject_length = []
  subject_word_count = []
  content_length = []
  content_word_count = []
  is_am_list = []
  is_weekday_list = []
  group_Networking = []
  group_Restraining = []
  group_Facilitating = []
  group_Motivating = []
  group_PhysicalAction = []
  group_Praise = []
  group_Criticism = []
  final_outcome = []

  cur_time_stamp = datetime.datetime.now()

  raw_text.append(text)
  group_Networking.append(words_groups[0])
  group_Restraining.append(words_groups[1])
  group_Facilitating.append(words_groups[2])
  group_Motivating.append(words_groups[3])
  group_PhysicalAction.append(words_groups[4])
  group_Praise.append(words_groups[5])
  group_Criticism.append(words_groups[6])
  outcome_tots = [words_groups[0], words_groups[1], words_groups[2], words_groups[3], words_groups[4], words_groups[5], words_groups[6]]
  final_outcome.append(outcome_tots.index(max(outcome_tots)))

  subject_length.append(len(subject_to_predict))
  subject_word_count.append(len(subject_to_predict.split()))
  content_length.append(len(content_to_predict))
  content_word_count.append(len(content_to_predict.split()))
  dt = cur_time_stamp
  is_am = 'no'
  if (dt.time() < datetime.time(12)): is_am = 'yes'
  is_am_list.append(is_am)
  is_weekday = 'no'
  if (dt.weekday() < 6): is_weekday = 'yes'
  is_weekday_list.append(is_weekday)

  # add simple engineered features?
  training_set = pd.DataFrame({
                "raw_text":raw_text,
                "group_Networking":group_Networking[0],
                "group_Restraining":group_Restraining[0],
                "group_Facilitating":group_Facilitating[0],
                "group_Motivating":group_Motivating[0],
                "group_PhysicalAction":group_PhysicalAction[0],
                "group_Praise":group_Praise[0],
                "group_Criticism":group_Criticism[0],
                "subject_length":subject_length,
                "subject_word_count":subject_word_count,
                "content_length":content_length,
                "content_word_count":content_word_count,
                "is_AM":is_am_list,
                "is_weekday":is_weekday_list,
                "outcome":final_outcome})


  return(training_set)

scrubbed_entry = scrub_text(subject_to_predict, content_to_predict, departments)

y_pred = classifier.predict(input_fn=lambda: input_fn(scrubbed_entry), as_iterable=False)
print(y_pred)

department_names = ['Networking', 'Restraining', 'Facilitating', 'Motivating', 'Physical Action', 'Praise', 'Criticism']

print('Forward request to: ' +  department_names[y_pred[0]])
