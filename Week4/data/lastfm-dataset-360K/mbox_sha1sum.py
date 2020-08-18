# Example to get the SHA1 sum, given a user's email
import hashlib

mbox = 'foo@bar.net'
mbox_sha1sum = hashlib.sha1(mbox)
print(mbox_sha1sum.hexdigest())
#'62e8a8e6a893103a75c2132fa880e7f07e8fa517'
