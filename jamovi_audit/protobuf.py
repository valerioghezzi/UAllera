"""Small bounded protobuf wire decoder. Field mapping: jamovi 2.7.4 jamovi.proto."""
import struct
MAX_MESSAGE=32*1024*1024
MAX_FIELDS=200000
MAX_DEPTH=64

def fields(data, depth=0):
    if depth>MAX_DEPTH or len(data)>MAX_MESSAGE:raise ValueError('Limite protobuf superato')
    result={};pos=0;count=0
    def varint():
        nonlocal pos
        n=0
        for shift in range(0,70,7):
            if pos>=len(data):raise ValueError('Varint protobuf troncato')
            b=data[pos];pos+=1;n|=(b&127)<<shift
            if b<128:
                if shift==63 and b>1:raise ValueError('Varint protobuf oltre 64 bit')
                return n
        raise ValueError('Varint protobuf non valido')
    while pos<len(data):
        count+=1
        if count>MAX_FIELDS:raise ValueError('Troppi campi protobuf')
        key=varint();num=key>>3;wire=key&7
        if num==0:raise ValueError('Campo protobuf zero')
        if wire==0:value=varint()
        elif wire in (1,5):
            size=8 if wire==1 else 4
            if pos+size>len(data):raise ValueError('Campo protobuf troncato')
            value=struct.unpack('<d' if wire==1 else '<f',data[pos:pos+size])[0];pos+=size
        elif wire==2:
            size=varint()
            if pos+size>len(data):raise ValueError('Messaggio protobuf troncato')
            value=data[pos:pos+size];pos+=size
        else:raise ValueError('Wire type protobuf non supportato')
        result.setdefault(num,[]).append((wire,value))
    return result

def val(f,n,default=None):return f[n][-1][1] if n in f else default

def string(f,n,default=''):
    v=val(f,n)
    if v is None:return default
    if not isinstance(v,bytes):raise ValueError('Tipo string protobuf non valido')
    return v.decode('utf-8')

def signed(v):return v-(1<<64) if v>=(1<<63) else v
