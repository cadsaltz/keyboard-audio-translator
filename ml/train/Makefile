
PROGRAM1 = trainx.py
TARGET1 = /usr/local/bin/trainx
PROGRAM2 = trainy.py
TARGET2 = /usr/local/bin/trainy
PROGRAM3 = listen.py
TARGET3 = /usr/local/bin/listen

BACKUP_DIR = /home/caden-saltzberg/projects/listen/bu
DATE = $(shell date +%m_%d)

isntall:
	sudo cp $(PROGRAM1) $(TARGET1)
	@echo "Installed $(PROGRAM1) as $(TARGET1)"
	sudo cp $(PROGRAM2) $(TARGET2)
	@echo "Installed $(PROGRAM2) as $(TARGET2)"
	sudo cp $(PROGRAM3) $(TARGET3)
	@echo "Installed $(PROGRAM3) as $(TARGET3)"

backup:
	cp $(PROGRAM1) $(BACKUP_DIR)/$(basename $(PROGRAM1))$(DATE).py
	@echo "Backup created: $(BACKUP_DIR)/$(basename $(PROGRAM1))$(DATE)"
	cp $(PROGRAM2) $(BACKUP_DIR)/$(basename $(PROGRAM2))$(DATE).py
	@echo "Backup created: $(BACKUP_DIR)/$(basename $(PROGRAM2))$(DATE)"
	cp $(PROGRAM3) $(BACKUP_DIR)/$(basename $(PROGRAM3))$(DATE).py
	@echo "Backup created: $(BACKUP_DIR)/$(basename $(PROGRAM3))$(DATE)"

all:
	install
	backup

