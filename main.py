import sys

def readFile(file, mode='r'):
    data = []
    reader = open(file, mode)
    for line in reader.readlines():
        for s in line.split():
            data.append(s)
    return data

def completeFront(s, l, c = ' '):
    while len(s) < l:
        s = c + s
    return s

def completeBack(s, l, c = ' '):
    while len(s) < l:
        s += c
    return s

def checkPattern(s, pattern, c='?'):
    if len(s) != len(pattern):
        return False
    for i in range(len(s)):
        if pattern[i] == c:
            continue
        if pattern[i] != s[i]:
            return False
    return True

def invertBinaryString(s):
    res = ''
    for c in s:
        if c == '0':
            res += '1'
        else:
            res += '0'
    return res

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
value = dict()

for i in range(len(digits)):
    value[digits[i]] = i

def ConvertFrom(x, a):
    ans = 0
    for d in x:
        ans = ans * a + value[d]
    return ans

def ConvertTo(x, a):
    ans = ""
    while x > 0:
        ans += digits[x % a]
        x //= a
    ans = ans[::-1]
    return ans

def numberSystemConvert(x, sfrom, sto):
    x10 = ConvertFrom(x, 2)
    return ConvertTo(x10, 16)

def twosComplementTo(x, l):
    if x >= 0:
        return completeFront("0" + ConvertTo(x, 2), l - 1, '0')
    res = completeFront(ConvertTo(-x, 2), l - 1, '0')
    res = invertBinaryString(res)
    for i in range(l - 2, -1, -1):
        if res[i] == '0':
            res = res[0:i] + '1'
            while len(res) + 1 < l:
                res += '0'
            return '1' + res

def twosComplementFrom(s):
    n = len(s)
    p = 1
    res = 0
    for i in range(n - 1, -1, -1):
        if i == 0:
            p *= -1
        if s[i] == '1':
            res += p
        p *= 2
    return res

class Byte:

    def __init__(self, x=None, s=None):
        if x != None:
            self.x = x
            self.s = completeFront(ConvertTo(x, 2), 8, '0')
        else:
            self.s = completeFront(s, 8, '0')
            self.x = ConvertFrom(s, 2)

    def __abs__(self):
        return self.x

    def __int__(self):
        return self.x

    def __str__(self):
        return self.s

    def __add__(self, other):
        return Byte((self.x + other.x) % 256)

    def __sub__(self, other):
        return Byte((self.x - other.x + 256) % 256)

    def __mul__(self, other):
        return Byte((self.x * other.x) % 256)

    def __divmod__(self, other):
        return Byte(self.x // other.x)

    def __eq__(self, other):
        if not (other.__class__ is Byte):
            other = Byte(other)
        return self.x == other.x

    def __lt__(self, other):
        return self.x < other.x

    def __gt__(self, other):
        return self.x > other.x

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def __hash__(self):
        return self.x

    def prefix(self, n):
        return Byte(self.x // (2 ** (8 - n)))

    def suffix(self, n):
        return Byte(self.x & (2 ** n - 1))

    def containsOpcode(self):
        x = self.suffix(7)
        return opcodes().__contains__(x)

    def __hex__(self):
        return hex(self.x)

    def reverse(self):
        self.s = self.s[::-1]


def opcodes():
    return [Byte(s=s) for s in readFile('files/opcodes')]


class Register:

    def __init__(self, number):
        self.number = number

    names = dict(
        [
            (0, 'zero'),
            (1, 'ra'),
            (2, 'sp'),
            (3, 'gp'),
            (4, 'tp'),
            (8, 's0/fp'),
            (9, 's1')
        ]
    )

    def __str__(self):
        if self.names.keys().__contains__(self.number):
            return self.names[self.number]
        if self.number <= 7:
            return "t" + str(self.number - 5)
        if self.number <= 17:
            return "a" + str(self.number - 10)
        if self.number <= 27:
            return "s" + str(self.number - 16)
        return "t" + str(self.number - 25)


class Instruction:

    def __init__(self, s=None, bytes=None, bigEndian=True):
        self.bigEndian = bigEndian
        self.littleEndian = not bigEndian
        if s != None:
            self.bytes = [Byte(s=s[i * 8:i * 8 + 7]) for i in range(4)]
            if self.littleEndian:
                self.bytes = self.bytes[::-1]
            self.s = ''.join((str(byte)) for byte in self.bytes)
        else:
            if self.littleEndian:
                bytes = bytes[::-1]
            self.bytes = bytes
            self.s = ''.join(str(byte) for byte in bytes)

    def __str__(self):
        return self.s

    def substr(self, l, r):
        return self.s[l:r]

    def part1(self):
        return self.substr(0, 7)

    def part2(self):
        return self.substr(7, 12)

    def part3(self):
        return self.substr(12, 17)

    def part4(self):
        return self.substr(17, 20)

    def part5(self):
        return self.substr(20, 25)

    def opcode(self):
        return self.substr(25, 32)

    def funct3(self):
        return self.part4()

    def rd(self):
        return self.part5()

    def imm7(self):
        return self.part1()

    def imm20(self):
        return self.part1() + self.part2() + self.part3() + self.part4()

    def rs2(self):
        return self.part2()

    def rs1(self):
        return self.part3()

    def imm5(self):
        return self.part5()

    def imm12(self):
        return self.part1() + self.part2()

    def shamt(self):
        return self.part2()

    I_R_TYPE_OPCODE = '0110011'
    I_I_TYPE_OPCODE = '0010011'
    I_I_TYPE_LOAD_OPCODE = '0000011'
    I_S_TYPE_OPCODE = '0100011'
    I_SB_TYPE_OPCODE = '1100011'
    M_OPCODE = '0110011'

    ADD_PATTERN = '0000000??????????000?????' + I_R_TYPE_OPCODE
    SUB_PATTERN = '0100000??????????000?????' + I_R_TYPE_OPCODE
    SLL_PATTERN = '0000000??????????001?????' + I_R_TYPE_OPCODE
    SLT_PATTERN = '0000000??????????010?????' + I_R_TYPE_OPCODE
    SLTU_PATTERN = '0000000??????????011?????' + I_R_TYPE_OPCODE
    XOR_PATTERN = '0000000??????????100?????' + I_R_TYPE_OPCODE
    SRL_PATTERN = '0000000??????????101?????' + I_R_TYPE_OPCODE
    SRA_PATTERN = '0100000??????????101?????' + I_R_TYPE_OPCODE
    OR_PATTERN = '0000000??????????110?????' + I_R_TYPE_OPCODE
    AND_PATTERN = '0000000??????????111?????' + I_R_TYPE_OPCODE

    ADDI_PATTERN = '?????????????????000?????' + I_I_TYPE_OPCODE
    SLTI_PATTERN = '?????????????????010?????' + I_I_TYPE_OPCODE
    SLTIU_PATTERN = '?????????????????011?????' + I_I_TYPE_OPCODE
    XORI_PATTERN = '?????????????????100?????' + I_I_TYPE_OPCODE
    ORI_PATTERN = '?????????????????110?????' + I_I_TYPE_OPCODE
    ANDI_PATTERN = '?????????????????111?????' + I_I_TYPE_OPCODE
    SLLI_PATTERN = '0000000??????????001?????' + I_I_TYPE_OPCODE
    SRLI_PATTERN = '0000000??????????101?????' + I_I_TYPE_OPCODE
    SRAI_PATTERN = '0100000??????????101?????' + I_I_TYPE_OPCODE

    LB_PATTERN = '?????????????????000?????' + I_I_TYPE_LOAD_OPCODE
    LH_PATTERN = '?????????????????001?????' + I_I_TYPE_LOAD_OPCODE
    LW_PATTERN = '?????????????????010?????' + I_I_TYPE_LOAD_OPCODE
    LBU_PATTERN = '?????????????????100?????' + I_I_TYPE_LOAD_OPCODE
    LHU_PATTERN = '?????????????????101?????' + I_I_TYPE_LOAD_OPCODE

    SB_PATTERN = '?????????????????000?????' + I_S_TYPE_OPCODE
    SH_PATTERN = '?????????????????001?????' + I_S_TYPE_OPCODE
    SW_PATTERN = '?????????????????010?????' + I_S_TYPE_OPCODE

    BEQ_PATTERN = '?????????????????000?????' + I_SB_TYPE_OPCODE
    BNE_PATTERN = '?????????????????001?????' + I_SB_TYPE_OPCODE
    BLT_PATTERN = '?????????????????100?????' + I_SB_TYPE_OPCODE
    BGE_PATTERN = '?????????????????101?????' + I_SB_TYPE_OPCODE
    BLTU_PATTERN = '?????????????????110?????' + I_SB_TYPE_OPCODE
    BGEU_PATTERN = '?????????????????111?????' + I_SB_TYPE_OPCODE

    LUI_PATTERN = '?????????????????????????0110111'
    AUIPC_PATTERN = '?????????????????????????0010111'

    JAL_PATTERN = '?????????????????????????1101111'
    JALR_PATTERN = '?????????????????????????1100111'

    FENCE_PATTERN = '?????????????????000?????0001111'
    ECALL_PATTERN = '00000000000000000000000001110011'
    EBREAK_PATTERN = '00000000000100000000000001110011'

    MUL_PATTERN = '0000001??????????000?????' + M_OPCODE
    MULH_PATTERN = '0000001??????????001?????' + M_OPCODE
    MULHSU_PATTERN = '0000001??????????010?????' + M_OPCODE
    MULHU_PATTERN = '0000001??????????011?????' + M_OPCODE
    DIV_PATTERN = '0000001??????????100?????' + M_OPCODE
    DIVU_PATTERN = '0000001??????????101?????' + M_OPCODE
    REM_PATTERN = '0000001??????????110?????' + M_OPCODE
    REMU_PATTERN = '0000001??????????111?????' + M_OPCODE

    patterns = dict(
        [
            (ADD_PATTERN, ['ADD', 'I_R']),
            (SUB_PATTERN, ['SUB', 'I_R']),
            (SLL_PATTERN, ['SLL', 'I_R']),
            (SLT_PATTERN, ['SLT', 'I_R']),
            (SLTU_PATTERN, ['SLTU', 'I_R']),
            (XOR_PATTERN, ['XOR', 'I_R']),
            (SRL_PATTERN, ['SRL', 'I_R']),
            (SRA_PATTERN, ['SRA', 'I_R']),
            (OR_PATTERN, ['OR', 'I_R']),
            (AND_PATTERN, ['AND', 'I_I']),
            (ADDI_PATTERN, ['ADDI', 'I_I']),
            (SLLI_PATTERN, ['SLLI', 'I_I']),
            (SLTI_PATTERN, ['SLTI', 'I_I']),
            (SLTIU_PATTERN, ['SLTIU', 'I_I']),
            (XORI_PATTERN, ['XORI', 'I_I']),
            (SRLI_PATTERN, ['SRLI', 'I_I']),
            (SRAI_PATTERN, ['SRAI', 'I_I']),
            (ORI_PATTERN, ['ORI', 'I_I']),
            (ANDI_PATTERN, ['ANDI', 'I_I']),
            (LB_PATTERN, ['LB', 'I_I_L']),
            (LH_PATTERN, ['LH', 'I_I_L']),
            (LW_PATTERN, ['LW', 'I_I_L']),
            (LBU_PATTERN, ['LBU', 'I_I_L']),
            (LHU_PATTERN, ['LHU', 'I_I_L']),
            (SB_PATTERN, ['SB', 'I_S']),
            (SH_PATTERN, ['SH', 'I_S']),
            (SW_PATTERN, ['SW', 'I_S']),
            (BEQ_PATTERN, ['BEQ', 'I_SB']),
            (BNE_PATTERN, ['BNE', 'I_SB']),
            (BLT_PATTERN, ['BLT', 'I_SB']),
            (BGE_PATTERN, ['BGE', 'I_SB']),
            (BLTU_PATTERN, ['BLTU', 'I_SB']),
            (BGEU_PATTERN, ['BGEU', 'I_SB']),
            (LUI_PATTERN, ['LUI', 'I_U']),
            (AUIPC_PATTERN, ['AUIPC', 'I_U']),
            (JAL_PATTERN, ['JAL', 'JAL']),
            (JALR_PATTERN, ['JALR', 'JALR']),
            (FENCE_PATTERN, ['FENCE', 'FENCE']),
            (ECALL_PATTERN, ['ECALL', 'NO_ARG']),
            (EBREAK_PATTERN, ['EBREAK', 'NO_ARG']),
            (MUL_PATTERN, ['MUL', 'M']),
            (MULH_PATTERN, ['MULH', 'M']),
            (MULHSU_PATTERN, ['MULHSU', 'M']),
            (MULHU_PATTERN, ['MULHU', 'M']),
            (DIV_PATTERN, ['DIV', 'M']),
            (DIVU_PATTERN, ['DIVU', 'M']),
            (REM_PATTERN, ['REM', 'M']),
            (REMU_PATTERN, ['REMU', 'M'])
        ]
    )

    def name(self):
        for pattern in self.patterns.keys():
            if checkPattern(self.s, pattern):
                return completeBack(self.patterns[pattern][0], 10, ' ').lower()

    def fname(self):
        for pattern in self.patterns.keys():
            if checkPattern(self.s, pattern):
                return self.patterns[pattern][1]

    def rdNumber(self):
        return Register(ConvertFrom(self.rd(), 2))

    def rs1Number(self):
        return Register(ConvertFrom(self.rs1(), 2))

    def rs2Number(self):
        return Register(ConvertFrom(self.rs2(), 2))


def I_R_FORMAT_DISASSEMBLER(instruction):
    rd = instruction.rdNumber()
    rs1 = instruction.rs1Number()
    rs2 = instruction.rs2Number()
    return instruction.name() + " " + str(rd) + ", " + str(rs1) + ", " + str(rs2)


def I_I_FORMAT_DISASSEMBLER(instruction):
    rd = instruction.rdNumber()
    rs1 = instruction.rs1Number()
    funct3 = instruction.funct3()
    imm = {'000', '010', '011', '100', '110', '111'}
    if imm.__contains__(funct3):
        x = twosComplementFrom(instruction.imm12())
        return instruction.name() + " " + str(rd) + ", " + str(rs1) + ", " + str(x)
    else:
        shamt = ConvertFrom(instruction.shamt(), 2)
        return instruction.name() + " " + str(rd) + ", " + str(rs1) + ", " + str(shamt)


def I_I_FORMAT_LOAD_DISASSEMBLER(instruction):
    rd = instruction.rdNumber()
    rs1 = instruction.rs1Number()
    imm12 = twosComplementFrom(instruction.imm12())
    return instruction.name() + " " + str(rd) + ", " + str(imm12) + "(" + str(rs1) + ")"


def I_S_FORMAT_DISASSEMBLER(instruction):
    rs1 = instruction.rs1Number()
    rs2 = instruction.rs2Number()
    imm7 = instruction.imm7()
    imm5 = instruction.imm5()
    imm12 = ConvertFrom(imm7 + imm5, 2)
    return instruction.name() + " " + str(rs2) + ", " + str(imm12) + "(" + str(rs1) + ")"


def I_SB_FORMAT_DISASSEMBLER(instruction):
    rs1 = instruction.rs1Number()
    rs2 = instruction.rs2Number()
    imm7 = instruction.imm7()
    imm5 = instruction.imm5()
    imm = ConvertFrom(imm7[0:1] + imm5[4:5] + imm7[1:7] + imm5[0:4], 2) * 2
    return instruction.name() + " " + str(rs2) + ", " + str(rs1) + ",offset = " + str(imm) + " bytes"


def I_U_FORMAT_DISASSEMBLER(instruction):
    rd = instruction.rdNumber()
    imm20 = instruction.imm20()
    return instruction.name() + " " + str(rd) + ", 0" + numberSystemConvert(imm20, 2, 16)


def JAL_DISASSEMBLER(instruction):
    rd = instruction.rdNumber()
    imm20 = instruction.imm20()
    s1 = imm20[0:1]
    s2 = imm20[1:11]
    s3 = imm20[11:12]
    s4 = imm20[12:20]
    imm = s1 + s4 + s3 + s2
    return instruction.name() + " " + str(rd) + ", pc + " + str(ConvertFrom(imm, 2))

def JAL_To(instruction):
    imm20 = instruction.imm20()
    s1 = imm20[0:1]
    s2 = imm20[1:11]
    s3 = imm20[11:12]
    s4 = imm20[12:20]
    imm = s1 + s4 + s3 + s2
    return ConvertFrom(imm, 2)

def JALR_DISASSEMBLER(instruction):
    rd = instruction.rdNumber()
    imm12 = instruction.imm12()
    base = instruction.rs1Number()
    return instruction.name() + " " + str(rd) + ", " + str(base) + ", " + str(ConvertFrom(imm12, 2))

def NO_ARG_INSTRUCTION_DISASSEMBLER(instruction):
    return instruction.name()


def M_FORMAT_DISASSEMBLER(instruction):
    rd = instruction.rdNumber()
    rs1 = instruction.rs1Number()
    rs2 = instruction.rs2Number()
    return instruction.name() + " " + str(rd) + "," + str(rs1) + "," + str(rs2)


def bytesToBlocks(bytes, blockSize):
    blocks = []
    block = []
    for byte in bytes:
        block.append(byte)
        if len(block) == blockSize:
            blocks.append(block.copy())
            block.clear()
    return blocks


disassemblers = dict(
    [
        ('I_R', I_R_FORMAT_DISASSEMBLER),
        ('I_I', I_I_FORMAT_DISASSEMBLER),
        ('I_I_L', I_I_FORMAT_LOAD_DISASSEMBLER),
        ('I_S', I_S_FORMAT_DISASSEMBLER),
        ('I_SB', I_SB_FORMAT_DISASSEMBLER),
        ('I_U', I_U_FORMAT_DISASSEMBLER),
        ('JAL', JAL_DISASSEMBLER),
        ('JALR', JALR_DISASSEMBLER),
        ('NO_ARG', NO_ARG_INSTRUCTION_DISASSEMBLER),
        ('M', M_FORMAT_DISASSEMBLER)
    ]
)


def disassemble(instruction):
    fname = instruction.fname()
    if fname == None:
        return "unknown command"
    f = disassemblers[fname]
    return [f(instruction), fname]


def disassembleBytes(bytes, offset, labelsNames, bigEndian=True, fout=None):
    blocks = bytesToBlocks(bytes, 4)
    n = len(blocks)
    disassembled = []
    labels = [None for i in range(n)]
    for i in range(n):
        instruction = Instruction(bytes=blocks[i], bigEndian=bigEndian)
        d = disassemble(instruction)
        if d[1] == 'JAL':
            to = JAL_To(instruction) + 4 * i
            labels[to // 4] = completeBack("<" + labelsNames[to + 4] + ">", 10, ' ')
        disassembled.append([completeFront(ConvertTo(offset + 4 * i, 16), 8, '0').lower() + ":", d[0]])
    for i in range(n):
        labelOrNothing = '          '
        if labels[i] != None:
            labelOrNothing = labels[i]
        if fout == None:
            print(disassembled[i][0], labelOrNothing, disassembled[i][1])
        else:
            print(disassembled[i][0], labelOrNothing, disassembled[i][1], file=fout)
    if fout != None:
        fout.close()

class ELFFile:

    def __init__(self, s=None, file=None):
        if s != None:
            self.s = s
            self.bytes = []
            for i in range(0, len(s), 8):
                self.bytes.append(Byte(s=self.s[i:i + 8]))
        else:
            try:
                fin = open(file, 'rb')
            except IOError:
                print("Can't open file")
                exit(0)
            self.bytes = []
            for s in fin.readlines():
                for byte in s:
                    self.bytes.append(Byte(x=byte))
            self.s = ''.join(str(byte) for byte in self.bytes)
        self.numberOfBytes = len(self.bytes)
        begin = self.bytes[0:4]
        elfBegin = [127, 69, 76, 70]
        for i in range(4):
            if begin[i] != elfBegin[i]:
                print("Unsupported format")
                exit(0)


    def __str__(self):
        return self.s

    LITTLE_ENDIAN = 1
    BIG_ENDIAN = 2

    def header(self):
        return self.bytes[0:52]

    def e_ident(self):
        return self.header()[0:16]

    def wordSize(self):
        return self.e_ident()[4]

    def endian(self):
        return self.e_ident()[5]

    def getField(self, l, r):
        res = self.header()[l:r]
        if self.endian() == self.LITTLE_ENDIAN:
            res = res[::-1]
        return ConvertFrom(''.join(str(byte) for byte in res), 2)

    def e_type(self):
        return self.getField(16, 18)

    def e_machine(self):
        return self.getField(18, 20)

    def e_version(self):
        return self.getField(20, 24)

    def e_entry(self):
        return self.getField(24, 28)

    def e_phoff(self):
        return self.getField(28, 32)

    def e_shoff(self):
        return self.getField(32, 36)

    def e_flags(self):
        return self.getField(36, 40)

    def e_ehsize(self):
        return self.getField(40, 42)

    def e_phentsize(self):
        return self.getField(42, 44)

    def e_phnum(self):
        return self.getField(44, 46)

    def e_shentsize(self):
        return self.getField(46, 48)

    def e_shnum(self):
        return self.getField(48, 50)

    def e_shstrndx(self):
        return self.getField(50, 52)

    def programBlock(self):
        shentsize = self.e_shentsize()
        shnum = self.e_shnum()
        r = self.numberOfBytes - shentsize * shnum
        return self.bytes[52:r]

    def program(self):
        programBlock = self.programBlock()
        blocks = bytesToBlocks(programBlock, 4)
        if self.endian() == self.LITTLE_ENDIAN:
            for i in range(len(blocks)):
                blocks[i] = blocks[i][::-1]
        return [Instruction(bytes=block) for block in blocks]

    def sectionsHeaders(self):
        shoff = self.e_shoff()
        shentsize = self.e_shentsize()
        shnum = self.e_shnum()
        headers = []
        for i in range(shnum):
            headers.append(SectionHeader(self.bytes[i * shentsize + shoff: (i + 1) * shentsize + shoff]))
        return headers

    def getSection(self, sectionHeader):
        offset = sectionHeader.sh_offset(self.endian() == self.BIG_ENDIAN)
        return self.bytes[offset:offset + sectionHeader.sh_size(self.endian() == self.BIG_ENDIAN)]


class SectionHeader:

    def __init__(self, bytes):
        self.bytes = bytes

    def getField(self, l, r, bigEndian=True):
        res = self.bytes[l:r]
        if not bigEndian:
            res = res[::-1]
        return ConvertFrom(''.join(str(byte) for byte in res), 2)

    def sh_name(self, bigEndian=True):
        return self.getField(0, 4, bigEndian)

    def sh_type(self, bigEndian=True):
        return self.getField(4, 8, bigEndian)

    SHT_NULL = 0
    SHT_PROGBITS = 1
    SHT_SYMTAB = 2
    SHT_STRTAB = 3

    types = dict(
        [
            (SHT_NULL, 'SHT_NULL'),
            (SHT_PROGBITS, 'SHT_PROGBITS'),
            (SHT_SYMTAB, 'SHT_SYMTAB'),
            (SHT_STRTAB, 'SHT_STRTAB'),
            (4, 'SHT_RELA'),
            (5, 'SHT_HASH'),
            (6, 'SHT_DYNAMIC'),
            (7, 'SHT_NOTE'),
            (8, 'SHT_NOBITS'),
            (9, 'SHT_REL'),
            (10, 'SHT_SHLIB'),
            (11, 'SHT_DYNSYM'),
            (14, 'INIT_ARRAY'),
            (15, 'FINI_ARRAY'),
            (16, 'PREINIT_ARRAY'),
            (6 * 16 ** 7, 'SHT_LOOS'),
            (7 * 16 ** 7 - 1, 'SHT_HIOS'),
            (7 * 16 ** 7, 'SHT_LOPROC'),
            (8 * 16 ** 7 - 1, 'SHT_HIPROC'),
            (8 * 16 ** 7, 'LOUSER'),
            (16 ** 8 - 1, 'HIUSER')
        ]
    )

    def shtypeName(self, bigEndian=True):
        shtype = self.sh_type(bigEndian)
        return self.types[shtype]

    def sh_flags(self, bigEndian=True):
        return self.getField(8, 12, bigEndian)

    flags = dict(
        [
            (1, 'SHF_WRITE'),
            (2, 'SHF_ALLOC'),
            (4, 'SHF_EXECINSTR'),
            (16, 'SHF_MERGE'),
            (32, 'SHF_STRINGS'),
            (64, 'SHF_INFO_LINK'),
            (128, 'SHF_LINK_ORDER'),
            (256, 'SHF_OS_NONCONFORMING'),
            (512, 'SHF_GROUP'),
            (255 * 16 ** 5, 'SHF_MASKOS'),
            (15 * 16 ** 7, 'SHF_MASKPROC')
        ]
    )

    def shflagsNames(self, bigEndian=True):
        flagsNames = []
        mask = self.sh_flags(bigEndian)
        for key in self.flags.keys():
            if (mask & key) > 0:
                flagsNames.append(self.flags[key])
        return flagsNames

    def sh_addr(self, bigEndian=True):
        return self.getField(12, 16, bigEndian)

    def sh_offset(self, bigEndian=True):
        return self.getField(16, 20, bigEndian)

    def sh_size(self, bigEndian=True):
        return self.getField(20, 24, bigEndian)

    def sh_link(self, bigEndian=True):
        return self.getField(24, 28, bigEndian)

    def sh_info(self, bigEndian=True):
        return self.getField(28, 32, bigEndian)

    def sh_addralign(self, bigEndian=True):
        return self.getField(32, 36, bigEndian)

    def sh_entsize(self, bigEndian=True):
        return self.getField(36, 40, bigEndian)

class SymTable:

    def __init__(self, bytes, bigEndian=True):
        self.bigEndian = bigEndian
        self.littleEndian = not bigEndian
        self.bytes = bytes
        self.n = len(self.bytes) // 16

    def getField(self, l, r):
        res = self.bytes[l:r]
        if self.littleEndian:
            res = res[::-1]
        return res

    def st_name(self, i):
        return self.getField(16 * i, 16 * i + 4)

    def st_value(self, i):
        return self.getField(16 * i + 4, 16 * i + 8)

    def st_size(self, i):
        return self.getField(16 * i + 8, 16 * i + 12)

    def st_shndx(self, i):
        return self.getField(16 * i + 12, 16 * i + 16)

    def getStNames(self):
        names = []
        for i in range(self.n):
            names.append(self.st_name(i))
        return names

    def getStValues(self):
        values = []
        for i in range(self.n):
            values.append(self.st_value(i))
        return values

class StrTable:

    def __init__(self, bytes):
        self.bytes = bytes

    def getName(self, i):
        name = ''
        while self.bytes[i] != 0:
            name += chr(self.bytes[i])
            i += 1
        return name

file = sys.argv[1]
fout = None

if len(sys.argv) == 3:
    fout = open(sys.argv[2], 'w')

elf = ELFFile(file=file)
endian = elf.endian() != elf.LITTLE_ENDIAN

labelsNames = dict()
labelsValues = []

for sectionHeader in elf.sectionsHeaders():
    type = sectionHeader.sh_type(endian)
    if type == SectionHeader.SHT_SYMTAB:
        symtab = SymTable(elf.getSection(sectionHeader), endian)
        for i in range(symtab.n):
            labelsNames[ConvertFrom(''.join(str(byte) for byte in symtab.st_value(i)),
                                    2)] = ConvertFrom(
                ''.join(str(byte) for byte in symtab.st_name(i)), 2)

for sectionHeader in elf.sectionsHeaders():
    type = sectionHeader.sh_type(endian)
    if type == SectionHeader.SHT_STRTAB:
        strTab = StrTable(elf.getSection(sectionHeader))
        for name in labelsNames.keys():
            labelsNames[name] = strTab.getName(labelsNames[name])

for sectionHeader in elf.sectionsHeaders():
    type = sectionHeader.sh_type(endian)
    if type == SectionHeader.SHT_PROGBITS:
        if sectionHeader.sh_flags(endian) == 6:
            code = elf.getSection(sectionHeader)
            disassembleBytes(code, sectionHeader.sh_offset(endian), labelsNames, bigEndian=False, fout=fout)