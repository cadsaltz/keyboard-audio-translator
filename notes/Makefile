

MAIN = guess.py
FUNCTION = mapping.py
TARGET = /usr/local/bin/guess
BACKUP_DIR = /home/caden-saltzberg/projects/listen/bu
DATE = $(shell date +%m-%d)

install:
	sudo cp $(MAIN) $(TARGET)
#	chmod +x $(TARGET)
	@echo "Installed $(MAIN) as $(TARGET)"

backup:
	cp $(MAIN) $(BACKUP_DIR)/$(basename $(MAIN))$(DATE).py
	@echo "Backup created: $(BACKUP_DIR)/$(basename $(MAIN))$(DATE).py"
	cp $(FUNCTION) $(BACKUP_DIR)/$(basename $(FUNCTION))$(DATE).py
	@echo "Backup created: $(BACKUP_DIR)/$(basename $(FUNCTION))$(DATE).py"

all: 
	install 
	backup
