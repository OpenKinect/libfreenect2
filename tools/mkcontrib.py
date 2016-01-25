# 1. Visual inspection: python mkcontrib.py
# 2. If OK, python mkcontrib.py | grep -v ^# > CONTRIB

from subprocess import Popen, PIPE
from collections import defaultdict, Counter

p = Popen(["git","log","--no-merges","--format=%aN <%aE>"], stdout=PIPE).stdout
common_email = defaultdict(Counter)
common_name = defaultdict(Counter)
for line in p:
    author, email = line.rstrip().split('<')
    common_email[author].update([email])
    common_name[email].update([author])

for email in common_name:
    names = common_name[email]
    names = sorted(names, key=lambda x: (names[x], len(x)), reverse=True)
    for name in names[1:]:
        common_email[names[0]] += common_email[name]
        del common_email[name]
        print '# Less common or shorter name', name, 'is replaced by', names[0]

for name in sorted(common_email):
    for k in common_email[name].most_common()[1:]:
        print '# Less common email <' + k[0] + ' is removed.'
    print name + '<' + common_email[name].most_common(1)[0][0]
