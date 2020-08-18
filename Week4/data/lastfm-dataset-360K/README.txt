===================
lastfm-dataset-360K
===================

Version 1.2
March 2010

. What is this?

    This dataset contains <user, artist, plays> tuples collected from Last.fm API ( http://www.last.fm/api ), 
    using the user.getTopArtists() method ( http://www.last.fm/api/show?service=300 )

. Data Format:

    The data is formatted one entry per line as follows (tab separated):

    usersha1-artmbid-artname-plays.tsv:
      user-mboxsha1 \t musicbrainz-artist-id \t artist-name \t plays

    usersha1-profile.tsv:
      user-mboxsha1 \t gender ('m'|'f'|empty) \t age (int|empty) \t country (str|empty) \t signup (date|empty)

. Example:

    usersha1-artmbid-artname-plays.tsv:
      000063d3fe1cf2ba248b9e3c3f0334845a27a6bf    af8e4cc5-ef54-458d-a194-7b210acf638f    cannibal corpse    48
      000063d3fe1cf2ba248b9e3c3f0334845a27a6bf    eaaee2c2-0851-43a2-84c8-0198135bc3a8    elis    31
      ...

    usersha1-profile.tsv
      000063d3fe1cf2ba248b9e3c3f0334845a27a6bf    m    19    Mexico    Apr 28, 2008
      ...

. Data Statistics:

     Total Lines:           17,559,530
     Unique Users:             359,347
     Artists with MBID:        186,642
     Artists without MBDID:    107,373

. Files:

    usersha1-artmbid-artname-plays.tsv (MD5: be672526eb7c69495c27ad27803148f1)
    usersha1-profile.tsv               (MD5: 51159d4edf6a92cb96f87768aa2be678)
    mbox_sha1sum.py                    (MD5: feb3485eace85f3ba62e324839e6ab39)

. License:

    The data in lastfm-dataset-360K is distributed with permission of Last.fm. 
    The data is made available for non-commercial use.
    Those interested in using the data or web services in a commercial context 
    should contact: partners [at] last [dot] fm. 
    For more information see http://www.last.fm/api/tos

. Acknowledgements:

    Thanks to Last.fm for providing the access to the <user,artist,plays> data via their
    web services. 
    Special thanks to Norman Casagrande.

. Contact:

    This data was collected by Oscar Celma. Send questions or comments to oscar.celma@upf.edu

