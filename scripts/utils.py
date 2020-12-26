def prepend_0s(s, l):
	"""
	s: string to be altered
	l: length to be made
	"""
	while (len(s)<l):
		s = '0' + s
	return s