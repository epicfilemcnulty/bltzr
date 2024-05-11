import json

SPECIAL_TOKENS = {
   256: '<BLOCK>',     #  Intended to enclose several other logically connected blocks,
   257: '</BLOCK>',    #  like <TXT> and/or <BIN>

   258: '<META>',      #  <TXT> and <BIN> can optionally include metadata:
   259: '</META>',     #       `<TXT><META>meta data</META>actual text data</TXT>`

   260: '<TXT>',       # Markers for text (i.e. UTF-8 valid) content
   261: '</TXT>',      #

   262: '<BIN>',       # Markers for binary content
   263: '</BIN>',      #

   264: '<CRON>',      # 
   265: '</CRON>',     #

   266: '<FN_CALL>',   # Function calls
   267: '</FN_CALL>',  #

   268: '<FN_RESP>',   # And function responses
   269: '</FN_RESP>',  #

   270: '<CHAT>',      # Markers for chat
   271: '</CHAT>',     #

   272: '<SYS>',       # System prompt/instruction
   273: '</SYS>',      #

   274: '<QUERY>',     # User queries
   275: '</QUERY>',    #

   276: '<REPLY>',     # LLM responses
   277: '</REPLY>',    #

   278: '<USR>',       # Reserved for future use =)
   279: '<PAD>'        # The last special token should always be the padding token
}

class Tokenizer:
    def __init__(self):
        self.special_tokens = SPECIAL_TOKENS
        self.vocab = list(range(256)) + list(self.special_tokens.keys())

    def get_token_id(self, token):
        token_list = list(self.special_tokens.values())
        for idx, t in enumerate(token_list):
            if t == token:
                return 256 + idx
        # If we can't find the token, we just return the last special token, i.e. <PAD> by default.
        return 255 + len(token_list) 

    def encode_simple(self, text, prefix_token=None, suffix_token=None):
        tokens = [b for b in text.encode()]
        if prefix_token is not None:
            tokens.insert(0, self.get_token_id(prefix_token))
        if suffix_token is not None:
            tokens.append(self.get_token_id(suffix_token))
        return tokens
                                                                                  
    def encode_meta(self, text):
        return self.encode_simple(text, prefix_token='<META>', suffix_token = '</META>')

    def encode_text(self, msg):
        meta = []
        if "meta" in msg:
            meta = self.encode_meta(msg['meta'])
        return [ self.get_token_id('<TXT>') ] + meta + [b for b in msg['content'].encode()] +  [ self.get_token_id('</TXT>') ]

    def encode_binary(self, binary):
        meta = []
        if "meta" in binary:
            meta = self.encode_meta(binary['meta'])
        return [ self.get_token_id('<BIN>') ] + meta + binary +  [ self.get_token_id('</BIN>') ]

    def encode(self, msgs, prefix_token=None, suffix_token=None):
        out = []
        if prefix_token:
            out = [ self.get_token_id(prefix_token) ]
        for msg in msgs:
            if msg['kind'] == 'spt': # SPecial Token
                out = out + [ self.get_token_id(msg['token']) ]
                if "content" in msg:
                    out = out + self.encode_simple(msg['content'])
            if msg['kind'] == 'txt':
                out = out + self.encode_text(msg)
            if msg['kind'] == 'bin':
                out = out + self.encode_binary(msg)
            if msg['kind'] == 'json':
                out = out + self.encode_simple(json.dumps(msg['content']))
        if suffix_token:
            out.append(self.get_token_id(suffix_token))
        return out

    def decode(self, output_ids, hide_special_tokens=False):
        output_tokens = []
        for token in output_ids:
            if token > 255:
               if not hide_special_tokens:
                   output_tokens.append(self.special_tokens[int(token)].encode())
            else:
               output_tokens.append(bytes([token]))
        # Convert tokens to bytes
        output_bytes = b''.join(output_tokens)
        # Decode output bytes
        output_text = output_bytes.decode(errors='ignore')
        return output_text
