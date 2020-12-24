# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from eva import Program, EvaProgram, Input, Output, evaluate, save, load, Type, Op, Term
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import numpy as np
from google.protobuf.json_format import MessageToJson
from google.protobuf import text_format
from google.protobuf.any_pb2 import Any
import known_type_pb2
import eva_pb2
import vec_lang_pb2 as vl


def decode_eva(eva_file, text_file):
    with open(eva_file, 'rb') as f, open(text_file, 'w') as g:
        read_kt = known_type_pb2.KnownType()
        read_kt.ParseFromString(f.read())
        read_eva = eva_pb2.Program()
        read_eva.ParseFromString(read_kt.contents.value)
        g.write(str(read_eva))


def decode_vl(vec_file, text_file):
    with open(vec_file, 'rb') as f, open(text_file, 'w') as g:
        read_vl = vl.Program()
        read_vl.ParseFromString(f.read())
        g.write(str(read_vl))


def translate_vec(name):
    vl_program = vl.Program()
    with open(name + '.vec', 'rb') as f:
        vl_program.ParseFromString(f.read())

    program = Program(name=name, vec_size=vl_program.vec_size)
    obj_to_term = {}

    # Translate Constants
    for c in vl_program.constants:
        if c.type == vl.VECTOR_CONST:
            elements = []
            for e in c.vec.elements:
                elements.append(e)
            obj_to_term[c.obj.id] = program._make_dense_constant(elements)
        elif c.type == vl.SCALAR_CONST:
            obj_to_term[c.obj.id] = program._make_uniform_constant(c.vec.elements[-1])
        else:
            print("Encountered unknown constant: " + str(c))
        # TODO: Set scale for an individual thing  (requires extending wrapper.cpp?)

    # Translate inputs
    for i in vl_program.inputs:
        if i.type == vl.VECTOR_CIPHER:
            obj_to_term[i.obj.id] = program._make_input('input_' + str(i.obj.id), Type.Cipher)
            # TODO: Set scale for each input individually (requires extending wrapper.cpp?)
            program.set_input_scales(int(i.scale))
        else:
            print("Unknown input type")

    # Translate Instructions
    vl_to_eva_op = {
        vl.UNDEFINED_OP: Op.Undef,
        vl.NEGATE: Op.Negate,
        vl.ADD: Op.Add,
        vl.SUB: Op.Sub,
        vl.MULTIPLY: Op.Mul,
        vl.SUM: Op.Add,
        vl.COPY: 'ERROR: COPY NOT SUPPORTED',
        vl.ROTATE_LEFT: Op.RotateLeftConst,
        vl.ROTATE_RIGHT: Op.RotateRightConst,
        vl.RELINEARIZE: Op.Relinearize,
        vl.MOD_SWITCH: Op.ModSwitch,
        vl.RESCALE: Op.Rescale,
        vl.NORMALIZE_SCALE: 'ERROR: NORMALIZE_SCALE NOT SUPPORTED'
    }
    for i in vl_program.insts:
        if (vl_to_eva_op[i.op_code] == Op.RotateLeftConst):
            # Find the constant value
            value = None
            for c in vl_program.constants:
                if c.obj.id == i.args[1].id:
                    value = c.vec.elements[-1]
            obj_to_term[i.output.id] = program._make_left_rotation(obj_to_term[i.args[0].id], int(value))
        elif (vl_to_eva_op[i.op_code] == Op.RotateRightConst):
            # Find the constant value
            value = None
            for c in vl_program.constants:
                if c.obj.id == i.args[1].id:
                    value = c.vec.elements[-1]
            obj_to_term[i.output.id] = program._make_right_rotation(obj_to_term[i.args[0].id], int(value))
        else:
            obj_to_term[i.output.id] = program._make_term(vl_to_eva_op[i.op_code], [obj_to_term[a.id] for a in i.args])
    # Translate outputs
    for i in vl_program.outputs:
        obj_to_term[i.obj.id] = program._make_output('output', obj_to_term[i.obj.id])
        # TODO: Set scale for each output individually (requires extending wrapper.cpp?)
        program.set_output_ranges(int(i.scale))

    # Save pre-compile program
    save(program, name + '_nocompile.eva')
    decode_eva(name + '_nocompile.eva', name + '_nocompile.txt')

    # Compile EVA Program (adds encoding, rescaling, etc)
    compiler = CKKSCompiler()
    program_compiled, params, signature = compiler.compile(program)

    # Save compiled EVA Program
    save(program_compiled, name + '.eva')
    decode_eva(name + '.eva', name + '.txt')
    save(params, name + '.evaparams')
    save(signature, name + '.evasignature')


def run_program(name):
    #################################################
    print('Key generation time')

    params = load(name + '.evaparams')

    public_ctx, secret_ctx = generate_keys(params)

    save(public_ctx, name + '.sealpublic')
    save(secret_ctx, name + '.sealsecret')

    #################################################
    print('Runtime on client')

    signature = load(name + '.evasignature')
    public_ctx = load(name + '.sealpublic')

    inputs = {
        'input_0': [0.5] * signature.vec_size
    }
    encInputs = public_ctx.encrypt(inputs, signature)

    save(encInputs, name + '_inputs.sealvals')

    #################################################
    print('Runtime on server')

    program = load(name + '.eva')
    public_ctx = load(name + '.sealpublic')
    encInputs = load(name + '_inputs.sealvals')

    encOutputs = public_ctx.execute(program, encInputs)

    save(encOutputs, name + '_outputs.sealvals')

    #################################################
    print('Back on client')

    program = load(name + '.eva')
    secret_ctx = load(name + '.sealsecret')
    encOutputs = load(name + '_outputs.sealvals')

    outputs = secret_ctx.decrypt(encOutputs, signature)

    reference = evaluate(program, inputs)
    print('Expected', reference)
    print('Got', outputs)
    print('MSE', valuation_mse(outputs, reference))


def main():
    name = 'small_mnist_hw'
    translate_vec(name)
    run_program(name)


if __name__ == "__main__":
    main()
