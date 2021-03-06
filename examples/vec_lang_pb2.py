# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vec_lang.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='vec_lang.proto',
  package='VecLang',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0evec_lang.proto\x12\x07VecLang\"\x14\n\x06Object\x12\n\n\x02id\x18\x01 \x01(\x04\"o\n\x0bInstruction\x12\x1f\n\x06output\x18\x01 \x01(\x0b\x32\x0f.VecLang.Object\x12 \n\x07op_code\x18\x02 \x01(\x0e\x32\x0f.VecLang.OpCode\x12\x1d\n\x04\x61rgs\x18\x03 \x03(\x0b\x32\x0f.VecLang.Object\"\x1a\n\x06Vector\x12\x10\n\x08\x65lements\x18\x01 \x03(\x01\"W\n\x05Input\x12\x1c\n\x03obj\x18\x01 \x01(\x0b\x32\x0f.VecLang.Object\x12!\n\x04type\x18\x02 \x01(\x0e\x32\x13.VecLang.ObjectType\x12\r\n\x05scale\x18\x03 \x01(\x01\"x\n\x08\x43onstant\x12\x1c\n\x03obj\x18\x01 \x01(\x0b\x32\x0f.VecLang.Object\x12!\n\x04type\x18\x02 \x01(\x0e\x32\x13.VecLang.ObjectType\x12\r\n\x05scale\x18\x03 \x01(\x01\x12\x1c\n\x03vec\x18\x04 \x01(\x0b\x32\x0f.VecLang.Vector\"5\n\x06Output\x12\x1c\n\x03obj\x18\x01 \x01(\x0b\x32\x0f.VecLang.Object\x12\r\n\x05scale\x18\x02 \x01(\x01\"\xa8\x01\n\x07Program\x12\x10\n\x08vec_size\x18\x01 \x01(\x04\x12$\n\tconstants\x18\x02 \x03(\x0b\x32\x11.VecLang.Constant\x12\x1e\n\x06inputs\x18\x03 \x03(\x0b\x32\x0e.VecLang.Input\x12 \n\x07outputs\x18\x04 \x03(\x0b\x32\x0f.VecLang.Output\x12#\n\x05insts\x18\x05 \x03(\x0b\x32\x14.VecLang.Instruction\"\xd3\x01\n\nSEALObject\x12/\n\tseal_type\x18\x01 \x01(\x0e\x32\x1c.VecLang.SEALObject.SEALType\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\"\x85\x01\n\x08SEALType\x12\x19\n\x15\x45NCRYPTION_PARAMETERS\x10\x00\x12\x0e\n\nCIPHERTEXT\x10\x01\x12\r\n\tPLAINTEXT\x10\x02\x12\x0e\n\nSECRET_KEY\x10\x03\x12\x0e\n\nPUBLIC_KEY\x10\x04\x12\x0f\n\x0bGALOIS_KEYS\x10\x05\x12\x0e\n\nRELIN_KEYS\x10\x06\"j\n\x1bProgramEncryptionParameters\x12\x32\n\x15\x65ncryption_parameters\x18\x01 \x01(\x0b\x32\x13.VecLang.SEALObject\x12\x17\n\x0fgalois_elements\x18\x02 \x03(\x04\"\x88\x01\n\nPublicKeys\x12\'\n\npublic_key\x18\x01 \x01(\x0b\x32\x13.VecLang.SEALObject\x12(\n\x0bgalois_keys\x18\x02 \x01(\x0b\x32\x13.VecLang.SEALObject\x12\'\n\nrelin_keys\x18\x03 \x01(\x0b\x32\x13.VecLang.SEALObject\"\x83\x01\n\x0bNamedValues\x12\x30\n\x06values\x18\x01 \x03(\x0b\x32 .VecLang.NamedValues.ValuesEntry\x1a\x42\n\x0bValuesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\"\n\x05value\x18\x02 \x01(\x0b\x32\x13.VecLang.SEALObject:\x02\x38\x01\"y\n\tInputData\x12\x1c\n\x03obj\x18\x01 \x01(\x0b\x32\x0f.VecLang.Object\x12\x1c\n\x03vec\x18\x02 \x01(\x0b\x32\x0f.VecLang.Vector\x12!\n\x04type\x18\x03 \x01(\x0e\x32\x13.VecLang.ObjectType\x12\r\n\x05scale\x18\x04 \x01(\x01\"2\n\x0cProgramInput\x12\"\n\x06inputs\x18\x01 \x03(\x0b\x32\x12.VecLang.InputData\"H\n\nOutputData\x12\x1c\n\x03obj\x18\x01 \x01(\x0b\x32\x0f.VecLang.Object\x12\x1c\n\x03vec\x18\x02 \x01(\x0b\x32\x0f.VecLang.Vector\"5\n\rProgramOutput\x12$\n\x07outputs\x18\x01 \x03(\x0b\x32\x13.VecLang.OutputData*\xbf\x01\n\x06OpCode\x12\x10\n\x0cUNDEFINED_OP\x10\x00\x12\n\n\x06NEGATE\x10\x01\x12\x07\n\x03\x41\x44\x44\x10\x02\x12\x07\n\x03SUB\x10\x03\x12\x0c\n\x08MULTIPLY\x10\x04\x12\x07\n\x03SUM\x10\x05\x12\x08\n\x04\x43OPY\x10\x06\x12\x0f\n\x0bROTATE_LEFT\x10\x07\x12\x10\n\x0cROTATE_RIGHT\x10\x08\x12\x0f\n\x0bRELINEARIZE\x10\t\x12\x0e\n\nMOD_SWITCH\x10\n\x12\x0b\n\x07RESCALE\x10\x0b\x12\x13\n\x0fNORMALIZE_SCALE\x10\x0c*\x8e\x01\n\nObjectType\x12\x12\n\x0eUNDEFINED_TYPE\x10\x00\x12\x10\n\x0cSCALAR_CONST\x10\x01\x12\x10\n\x0cSCALAR_PLAIN\x10\x02\x12\x11\n\rSCALAR_CIPHER\x10\x03\x12\x10\n\x0cVECTOR_CONST\x10\x04\x12\x10\n\x0cVECTOR_PLAIN\x10\x05\x12\x11\n\rVECTOR_CIPHER\x10\x06\x62\x06proto3'
)

_OPCODE = _descriptor.EnumDescriptor(
  name='OpCode',
  full_name='VecLang.OpCode',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED_OP', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='NEGATE', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ADD', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SUB', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MULTIPLY', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SUM', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='COPY', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ROTATE_LEFT', index=7, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ROTATE_RIGHT', index=8, number=8,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='RELINEARIZE', index=9, number=9,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MOD_SWITCH', index=10, number=10,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='RESCALE', index=11, number=11,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='NORMALIZE_SCALE', index=12, number=12,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1527,
  serialized_end=1718,
)
_sym_db.RegisterEnumDescriptor(_OPCODE)

OpCode = enum_type_wrapper.EnumTypeWrapper(_OPCODE)
_OBJECTTYPE = _descriptor.EnumDescriptor(
  name='ObjectType',
  full_name='VecLang.ObjectType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED_TYPE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SCALAR_CONST', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SCALAR_PLAIN', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SCALAR_CIPHER', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='VECTOR_CONST', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='VECTOR_PLAIN', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='VECTOR_CIPHER', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1721,
  serialized_end=1863,
)
_sym_db.RegisterEnumDescriptor(_OBJECTTYPE)

ObjectType = enum_type_wrapper.EnumTypeWrapper(_OBJECTTYPE)
UNDEFINED_OP = 0
NEGATE = 1
ADD = 2
SUB = 3
MULTIPLY = 4
SUM = 5
COPY = 6
ROTATE_LEFT = 7
ROTATE_RIGHT = 8
RELINEARIZE = 9
MOD_SWITCH = 10
RESCALE = 11
NORMALIZE_SCALE = 12
UNDEFINED_TYPE = 0
SCALAR_CONST = 1
SCALAR_PLAIN = 2
SCALAR_CIPHER = 3
VECTOR_CONST = 4
VECTOR_PLAIN = 5
VECTOR_CIPHER = 6


_SEALOBJECT_SEALTYPE = _descriptor.EnumDescriptor(
  name='SEALType',
  full_name='VecLang.SEALObject.SEALType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ENCRYPTION_PARAMETERS', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='CIPHERTEXT', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PLAINTEXT', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SECRET_KEY', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PUBLIC_KEY', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='GALOIS_KEYS', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='RELIN_KEYS', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=706,
  serialized_end=839,
)
_sym_db.RegisterEnumDescriptor(_SEALOBJECT_SEALTYPE)


_OBJECT = _descriptor.Descriptor(
  name='Object',
  full_name='VecLang.Object',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='VecLang.Object.id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=27,
  serialized_end=47,
)


_INSTRUCTION = _descriptor.Descriptor(
  name='Instruction',
  full_name='VecLang.Instruction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='output', full_name='VecLang.Instruction.output', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='op_code', full_name='VecLang.Instruction.op_code', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='args', full_name='VecLang.Instruction.args', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=49,
  serialized_end=160,
)


_VECTOR = _descriptor.Descriptor(
  name='Vector',
  full_name='VecLang.Vector',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='elements', full_name='VecLang.Vector.elements', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=162,
  serialized_end=188,
)


_INPUT = _descriptor.Descriptor(
  name='Input',
  full_name='VecLang.Input',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='obj', full_name='VecLang.Input.obj', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='VecLang.Input.type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scale', full_name='VecLang.Input.scale', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=190,
  serialized_end=277,
)


_CONSTANT = _descriptor.Descriptor(
  name='Constant',
  full_name='VecLang.Constant',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='obj', full_name='VecLang.Constant.obj', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='VecLang.Constant.type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scale', full_name='VecLang.Constant.scale', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vec', full_name='VecLang.Constant.vec', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=279,
  serialized_end=399,
)


_OUTPUT = _descriptor.Descriptor(
  name='Output',
  full_name='VecLang.Output',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='obj', full_name='VecLang.Output.obj', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scale', full_name='VecLang.Output.scale', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=401,
  serialized_end=454,
)


_PROGRAM = _descriptor.Descriptor(
  name='Program',
  full_name='VecLang.Program',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='vec_size', full_name='VecLang.Program.vec_size', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='constants', full_name='VecLang.Program.constants', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='VecLang.Program.inputs', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='outputs', full_name='VecLang.Program.outputs', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='insts', full_name='VecLang.Program.insts', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=457,
  serialized_end=625,
)


_SEALOBJECT = _descriptor.Descriptor(
  name='SEALObject',
  full_name='VecLang.SEALObject',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='seal_type', full_name='VecLang.SEALObject.seal_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data', full_name='VecLang.SEALObject.data', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SEALOBJECT_SEALTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=628,
  serialized_end=839,
)


_PROGRAMENCRYPTIONPARAMETERS = _descriptor.Descriptor(
  name='ProgramEncryptionParameters',
  full_name='VecLang.ProgramEncryptionParameters',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='encryption_parameters', full_name='VecLang.ProgramEncryptionParameters.encryption_parameters', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='galois_elements', full_name='VecLang.ProgramEncryptionParameters.galois_elements', index=1,
      number=2, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=841,
  serialized_end=947,
)


_PUBLICKEYS = _descriptor.Descriptor(
  name='PublicKeys',
  full_name='VecLang.PublicKeys',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='public_key', full_name='VecLang.PublicKeys.public_key', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='galois_keys', full_name='VecLang.PublicKeys.galois_keys', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='relin_keys', full_name='VecLang.PublicKeys.relin_keys', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=950,
  serialized_end=1086,
)


_NAMEDVALUES_VALUESENTRY = _descriptor.Descriptor(
  name='ValuesEntry',
  full_name='VecLang.NamedValues.ValuesEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='VecLang.NamedValues.ValuesEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='VecLang.NamedValues.ValuesEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1154,
  serialized_end=1220,
)

_NAMEDVALUES = _descriptor.Descriptor(
  name='NamedValues',
  full_name='VecLang.NamedValues',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='values', full_name='VecLang.NamedValues.values', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_NAMEDVALUES_VALUESENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1089,
  serialized_end=1220,
)


_INPUTDATA = _descriptor.Descriptor(
  name='InputData',
  full_name='VecLang.InputData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='obj', full_name='VecLang.InputData.obj', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vec', full_name='VecLang.InputData.vec', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='VecLang.InputData.type', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scale', full_name='VecLang.InputData.scale', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=1222,
  serialized_end=1343,
)


_PROGRAMINPUT = _descriptor.Descriptor(
  name='ProgramInput',
  full_name='VecLang.ProgramInput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='inputs', full_name='VecLang.ProgramInput.inputs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=1345,
  serialized_end=1395,
)


_OUTPUTDATA = _descriptor.Descriptor(
  name='OutputData',
  full_name='VecLang.OutputData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='obj', full_name='VecLang.OutputData.obj', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vec', full_name='VecLang.OutputData.vec', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=1397,
  serialized_end=1469,
)


_PROGRAMOUTPUT = _descriptor.Descriptor(
  name='ProgramOutput',
  full_name='VecLang.ProgramOutput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='outputs', full_name='VecLang.ProgramOutput.outputs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=1471,
  serialized_end=1524,
)

_INSTRUCTION.fields_by_name['output'].message_type = _OBJECT
_INSTRUCTION.fields_by_name['op_code'].enum_type = _OPCODE
_INSTRUCTION.fields_by_name['args'].message_type = _OBJECT
_INPUT.fields_by_name['obj'].message_type = _OBJECT
_INPUT.fields_by_name['type'].enum_type = _OBJECTTYPE
_CONSTANT.fields_by_name['obj'].message_type = _OBJECT
_CONSTANT.fields_by_name['type'].enum_type = _OBJECTTYPE
_CONSTANT.fields_by_name['vec'].message_type = _VECTOR
_OUTPUT.fields_by_name['obj'].message_type = _OBJECT
_PROGRAM.fields_by_name['constants'].message_type = _CONSTANT
_PROGRAM.fields_by_name['inputs'].message_type = _INPUT
_PROGRAM.fields_by_name['outputs'].message_type = _OUTPUT
_PROGRAM.fields_by_name['insts'].message_type = _INSTRUCTION
_SEALOBJECT.fields_by_name['seal_type'].enum_type = _SEALOBJECT_SEALTYPE
_SEALOBJECT_SEALTYPE.containing_type = _SEALOBJECT
_PROGRAMENCRYPTIONPARAMETERS.fields_by_name['encryption_parameters'].message_type = _SEALOBJECT
_PUBLICKEYS.fields_by_name['public_key'].message_type = _SEALOBJECT
_PUBLICKEYS.fields_by_name['galois_keys'].message_type = _SEALOBJECT
_PUBLICKEYS.fields_by_name['relin_keys'].message_type = _SEALOBJECT
_NAMEDVALUES_VALUESENTRY.fields_by_name['value'].message_type = _SEALOBJECT
_NAMEDVALUES_VALUESENTRY.containing_type = _NAMEDVALUES
_NAMEDVALUES.fields_by_name['values'].message_type = _NAMEDVALUES_VALUESENTRY
_INPUTDATA.fields_by_name['obj'].message_type = _OBJECT
_INPUTDATA.fields_by_name['vec'].message_type = _VECTOR
_INPUTDATA.fields_by_name['type'].enum_type = _OBJECTTYPE
_PROGRAMINPUT.fields_by_name['inputs'].message_type = _INPUTDATA
_OUTPUTDATA.fields_by_name['obj'].message_type = _OBJECT
_OUTPUTDATA.fields_by_name['vec'].message_type = _VECTOR
_PROGRAMOUTPUT.fields_by_name['outputs'].message_type = _OUTPUTDATA
DESCRIPTOR.message_types_by_name['Object'] = _OBJECT
DESCRIPTOR.message_types_by_name['Instruction'] = _INSTRUCTION
DESCRIPTOR.message_types_by_name['Vector'] = _VECTOR
DESCRIPTOR.message_types_by_name['Input'] = _INPUT
DESCRIPTOR.message_types_by_name['Constant'] = _CONSTANT
DESCRIPTOR.message_types_by_name['Output'] = _OUTPUT
DESCRIPTOR.message_types_by_name['Program'] = _PROGRAM
DESCRIPTOR.message_types_by_name['SEALObject'] = _SEALOBJECT
DESCRIPTOR.message_types_by_name['ProgramEncryptionParameters'] = _PROGRAMENCRYPTIONPARAMETERS
DESCRIPTOR.message_types_by_name['PublicKeys'] = _PUBLICKEYS
DESCRIPTOR.message_types_by_name['NamedValues'] = _NAMEDVALUES
DESCRIPTOR.message_types_by_name['InputData'] = _INPUTDATA
DESCRIPTOR.message_types_by_name['ProgramInput'] = _PROGRAMINPUT
DESCRIPTOR.message_types_by_name['OutputData'] = _OUTPUTDATA
DESCRIPTOR.message_types_by_name['ProgramOutput'] = _PROGRAMOUTPUT
DESCRIPTOR.enum_types_by_name['OpCode'] = _OPCODE
DESCRIPTOR.enum_types_by_name['ObjectType'] = _OBJECTTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Object = _reflection.GeneratedProtocolMessageType('Object', (_message.Message,), {
  'DESCRIPTOR' : _OBJECT,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.Object)
  })
_sym_db.RegisterMessage(Object)

Instruction = _reflection.GeneratedProtocolMessageType('Instruction', (_message.Message,), {
  'DESCRIPTOR' : _INSTRUCTION,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.Instruction)
  })
_sym_db.RegisterMessage(Instruction)

Vector = _reflection.GeneratedProtocolMessageType('Vector', (_message.Message,), {
  'DESCRIPTOR' : _VECTOR,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.Vector)
  })
_sym_db.RegisterMessage(Vector)

Input = _reflection.GeneratedProtocolMessageType('Input', (_message.Message,), {
  'DESCRIPTOR' : _INPUT,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.Input)
  })
_sym_db.RegisterMessage(Input)

Constant = _reflection.GeneratedProtocolMessageType('Constant', (_message.Message,), {
  'DESCRIPTOR' : _CONSTANT,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.Constant)
  })
_sym_db.RegisterMessage(Constant)

Output = _reflection.GeneratedProtocolMessageType('Output', (_message.Message,), {
  'DESCRIPTOR' : _OUTPUT,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.Output)
  })
_sym_db.RegisterMessage(Output)

Program = _reflection.GeneratedProtocolMessageType('Program', (_message.Message,), {
  'DESCRIPTOR' : _PROGRAM,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.Program)
  })
_sym_db.RegisterMessage(Program)

SEALObject = _reflection.GeneratedProtocolMessageType('SEALObject', (_message.Message,), {
  'DESCRIPTOR' : _SEALOBJECT,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.SEALObject)
  })
_sym_db.RegisterMessage(SEALObject)

ProgramEncryptionParameters = _reflection.GeneratedProtocolMessageType('ProgramEncryptionParameters', (_message.Message,), {
  'DESCRIPTOR' : _PROGRAMENCRYPTIONPARAMETERS,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.ProgramEncryptionParameters)
  })
_sym_db.RegisterMessage(ProgramEncryptionParameters)

PublicKeys = _reflection.GeneratedProtocolMessageType('PublicKeys', (_message.Message,), {
  'DESCRIPTOR' : _PUBLICKEYS,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.PublicKeys)
  })
_sym_db.RegisterMessage(PublicKeys)

NamedValues = _reflection.GeneratedProtocolMessageType('NamedValues', (_message.Message,), {

  'ValuesEntry' : _reflection.GeneratedProtocolMessageType('ValuesEntry', (_message.Message,), {
    'DESCRIPTOR' : _NAMEDVALUES_VALUESENTRY,
    '__module__' : 'vec_lang_pb2'
    # @@protoc_insertion_point(class_scope:VecLang.NamedValues.ValuesEntry)
    })
  ,
  'DESCRIPTOR' : _NAMEDVALUES,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.NamedValues)
  })
_sym_db.RegisterMessage(NamedValues)
_sym_db.RegisterMessage(NamedValues.ValuesEntry)

InputData = _reflection.GeneratedProtocolMessageType('InputData', (_message.Message,), {
  'DESCRIPTOR' : _INPUTDATA,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.InputData)
  })
_sym_db.RegisterMessage(InputData)

ProgramInput = _reflection.GeneratedProtocolMessageType('ProgramInput', (_message.Message,), {
  'DESCRIPTOR' : _PROGRAMINPUT,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.ProgramInput)
  })
_sym_db.RegisterMessage(ProgramInput)

OutputData = _reflection.GeneratedProtocolMessageType('OutputData', (_message.Message,), {
  'DESCRIPTOR' : _OUTPUTDATA,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.OutputData)
  })
_sym_db.RegisterMessage(OutputData)

ProgramOutput = _reflection.GeneratedProtocolMessageType('ProgramOutput', (_message.Message,), {
  'DESCRIPTOR' : _PROGRAMOUTPUT,
  '__module__' : 'vec_lang_pb2'
  # @@protoc_insertion_point(class_scope:VecLang.ProgramOutput)
  })
_sym_db.RegisterMessage(ProgramOutput)


_NAMEDVALUES_VALUESENTRY._options = None
# @@protoc_insertion_point(module_scope)
