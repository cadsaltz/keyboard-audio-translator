
PROGRAM = listen.py
TARGET = /usr/local/bin/listen
BACKUP_DIR = /home/caden-saltzberg/projects/listen/bu
DATE = $(shell date +%m_%d)

isntall:
	sudo cp $(PROGRAM) $(TARGET)
	@echo "Installed $(PROGRAM) as $(TARGET)"

backup:
	cp $(PROGRAM) $(BACKUP_DIR)/$(basename $(PROGRAM))$(DATE).py
	@echo "Backup created: $(BACKUP_DIR)/$(basename $(PROGRAM))$(DATE)"

all:
	install
	backup

