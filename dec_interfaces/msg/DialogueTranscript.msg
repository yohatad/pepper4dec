# DialogueTranscript.msg
#
# Topic: /remote_dialogue/transcript  (remote_dialogue node)
#
# One transcript line from the remote voice server. A reply arrives as
# several "assistant" messages with the same turn_id, one per sentence.

# "user" (what the server heard) or "assistant" (one sentence of the reply)
string role
string text
# 0 for the welcome line, then one per visitor message
int32 turn_id
