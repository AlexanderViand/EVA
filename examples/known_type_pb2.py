# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: known_type.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='known_type.proto',
  package='eva.msg',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10known_type.proto\x12\x07\x65va.msg\x1a\x19google/protobuf/any.proto\"D\n\tKnownType\x12&\n\x08\x63ontents\x18\x01 \x01(\x0b\x32\x14.google.protobuf.Any\x12\x0f\n\x07\x63reator\x18\x02 \x01(\tb\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_any__pb2.DESCRIPTOR,])




_KNOWNTYPE = _descriptor.Descriptor(
  name='KnownType',
  full_name='eva.msg.KnownType',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='contents', full_name='eva.msg.KnownType.contents', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='creator', full_name='eva.msg.KnownType.creator', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=56,
  serialized_end=124,
)

_KNOWNTYPE.fields_by_name['contents'].message_type = google_dot_protobuf_dot_any__pb2._ANY
DESCRIPTOR.message_types_by_name['KnownType'] = _KNOWNTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

KnownType = _reflection.GeneratedProtocolMessageType('KnownType', (_message.Message,), {
  'DESCRIPTOR' : _KNOWNTYPE,
  '__module__' : 'known_type_pb2'
  # @@protoc_insertion_point(class_scope:eva.msg.KnownType)
  })
_sym_db.RegisterMessage(KnownType)


# @@protoc_insertion_point(module_scope)