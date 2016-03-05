#define FRAME_HEIGHT 720
#define FRAME_WIDTH 1280
#define FRAME_INTERVAL (1000/30)
#define PACK_SIZE 4096 //udp pack size; note that OSX limits < 8100 bytes
#define ENCODE_QUALITY 80
#define SERVER_ADDRESS "127.0.0.1" // Server IP adress
#define SERVER_PORT "10000"        // Server Port
#define MAX_FRAME_ID 30000 // max number of recorded frames, 16.6min max at 30 FPS (max frame ID sort of hardcoded in image naming too, see below)
