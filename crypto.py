import numpy as np
import struct

def print_binary(m):
	if isinstance(m, str):
		print(' '.join(f'{ord(byte):08b}' for byte in m))
	else:
		print(' '.join(f'{byte:08b}' for byte in m))

def print_string(m):
	if isinstance(m, str):
		print(''.join(byte for byte in m))
	else:
		print(''.join(chr(byte) for byte in m))

def crypt(key, message):
	reverse = []
	for i, lettre in enumerate(message.encode('utf8')):
		reverse.append( np.uint8(key[i % len(key) ]) ^ np.uint8(lettre))
	return reverse

if __name__ == "__main__":
	key = np.random.randint(0, 255, size=(8, ), dtype=np.uint8)
	print("Clé: ")
	print_binary(key)

	message = "Bonjour cryptographie!"
	print("Message: ")
	print_binary(message)



	print("Crypt: ")
	print_binary(crypt(key, message))
	# print(' '.join(f'{byte:08b}' for byte in crypt))

	print(2^3)