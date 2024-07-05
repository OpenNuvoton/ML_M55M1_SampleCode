# 1 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 407 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "../../../../Library/StdDriver/src/npu/ethosu_driver.c" 2
# 23 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
# 1 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h" 1
# 26 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
# 1 "../../../../Library/StdDriver/inc/npu\\ethosu_types.h" 1
# 26 "../../../../Library/StdDriver/inc/npu\\ethosu_types.h"
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdint.h" 1 3
# 56 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdint.h" 3
typedef signed char int8_t;
typedef signed short int int16_t;
typedef signed int int32_t;
typedef signed long long int int64_t;


typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long int uint64_t;





typedef signed char int_least8_t;
typedef signed short int int_least16_t;
typedef signed int int_least32_t;
typedef signed long long int int_least64_t;


typedef unsigned char uint_least8_t;
typedef unsigned short int uint_least16_t;
typedef unsigned int uint_least32_t;
typedef unsigned long long int uint_least64_t;




typedef signed int int_fast8_t;
typedef signed int int_fast16_t;
typedef signed int int_fast32_t;
typedef signed long long int int_fast64_t;


typedef unsigned int uint_fast8_t;
typedef unsigned int uint_fast16_t;
typedef unsigned int uint_fast32_t;
typedef unsigned long long int uint_fast64_t;






typedef signed int intptr_t;
typedef unsigned int uintptr_t;



typedef signed long long intmax_t;
typedef unsigned long long uintmax_t;
# 27 "../../../../Library/StdDriver/inc/npu\\ethosu_types.h" 2





enum ethosu_error_codes
{
    ETHOSU_SUCCESS = 0,
    ETHOSU_GENERIC_FAILURE = -1,
    ETHOSU_INVALID_PARAM = -2
};

enum ethosu_clock_q_request
{
    ETHOSU_CLOCK_Q_DISABLE = 0,
    ETHOSU_CLOCK_Q_ENABLE = 1,
    ETHOSU_CLOCK_Q_UNCHANGED = 2
};

enum ethosu_power_q_request
{
    ETHOSU_POWER_Q_DISABLE = 0,
    ETHOSU_POWER_Q_ENABLE = 1,
    ETHOSU_POWER_Q_UNCHANGED = 2
};

struct ethosu_id
{
    uint32_t version_status;
    uint32_t version_minor;
    uint32_t version_major;
    uint32_t product_major;
    uint32_t arch_patch_rev;
    uint32_t arch_minor_rev;
    uint32_t arch_major_rev;
};

struct ethosu_config
{
    uint32_t macs_per_cc;
    uint32_t cmd_stream_version;
    uint32_t custom_dma;
};

struct ethosu_hw_info
{
    struct ethosu_id version;
    struct ethosu_config cfg;
};
# 27 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h" 2

# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdbool.h" 1 3
# 29 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h" 2
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stddef.h" 1 3
# 38 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stddef.h" 3
  typedef signed int ptrdiff_t;
# 53 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stddef.h" 3
    typedef unsigned int size_t;
# 71 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stddef.h" 3
      typedef unsigned short wchar_t;
# 95 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stddef.h" 3
  typedef long double max_align_t;
# 30 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h" 2
# 49 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
struct ethosu_device;

enum ethosu_job_state
{
    ETHOSU_JOB_IDLE = 0,
    ETHOSU_JOB_RUNNING,
    ETHOSU_JOB_DONE
};

struct ethosu_job
{
    volatile enum ethosu_job_state state;
    const void *custom_data_ptr;
    int custom_data_size;
    const uint64_t *base_addr;
    const size_t *base_addr_size;
    int num_base_addr;
    void *user_arg;
};

struct ethosu_driver
{
    struct ethosu_device *dev;
    struct ethosu_driver *next;
    struct ethosu_job job;
    void *semaphore;
    uint64_t fast_memory;
    size_t fast_memory_size;
    uint32_t power_request_counter;
    _Bool status_error;
    _Bool reserved;
};

struct ethosu_driver_version
{
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
};

enum ethosu_request_clients
{
    ETHOSU_PMU_REQUEST = 0,
    ETHOSU_INFERENCE_REQUEST = 1,
};
# 104 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
void ethosu_irq_handler(struct ethosu_driver *drv);
# 115 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
void ethosu_flush_dcache(uint32_t *p, size_t bytes);
# 126 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
void ethosu_invalidate_dcache(uint32_t *p, size_t bytes);







void *ethosu_mutex_create(void);







void *ethosu_semaphore_create(void);







int ethosu_mutex_lock(void *mutex);







int ethosu_mutex_unlock(void *mutex);







int ethosu_semaphore_take(void *sem);







int ethosu_semaphore_give(void *sem);







void ethosu_inference_begin(struct ethosu_driver *drv, void *user_arg);







void ethosu_inference_end(struct ethosu_driver *drv, void *user_arg);
# 200 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
uint64_t ethosu_address_remap(uint64_t address, int index);
# 217 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
int ethosu_init(struct ethosu_driver *drv,
                const void *base_address,
                const void *fast_memory,
                const size_t fast_memory_size,
                uint32_t secure_enable,
                uint32_t privilege_enable);






void ethosu_deinit(struct ethosu_driver *drv);







int ethosu_soft_reset(struct ethosu_driver *drv);
# 247 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
int ethosu_request_power(struct ethosu_driver *drv);







void ethosu_release_power(struct ethosu_driver *drv);






void ethosu_get_driver_version(struct ethosu_driver_version *ver);







void ethosu_get_hw_info(struct ethosu_driver *drv, struct ethosu_hw_info *hw);
# 285 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
int ethosu_invoke_v3(struct ethosu_driver *drv,
                     const void *custom_data_ptr,
                     const int custom_data_size,
                     const uint64_t *base_addr,
                     const size_t *base_addr_size,
                     const int num_base_addr,
                     void *user_arg);
# 302 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
int ethosu_invoke_async(struct ethosu_driver *drv,
                        const void *custom_data_ptr,
                        const int custom_data_size,
                        const uint64_t *base_addr,
                        const size_t *base_addr_size,
                        const int num_base_addr,
                        void *user_arg);
# 319 "../../../../Library/StdDriver/inc/npu\\ethosu_driver.h"
int ethosu_wait(struct ethosu_driver *drv, _Bool block);







struct ethosu_driver *ethosu_reserve_driver(void);






void ethosu_release_driver(struct ethosu_driver *drv);






static inline int ethosu_invoke_v2(const void *custom_data_ptr,
                                   const int custom_data_size,
                                   const uint64_t *base_addr,
                                   const size_t *base_addr_size,
                                   const int num_base_addr)
{
    struct ethosu_driver *drv = ethosu_reserve_driver();
    int result = ethosu_invoke_v3(drv, custom_data_ptr, custom_data_size, base_addr, base_addr_size, num_base_addr, 0);
    ethosu_release_driver(drv);
    return result;
}
# 24 "../../../../Library/StdDriver/src/npu/ethosu_driver.c" 2
# 1 "../../../../Library/StdDriver/src/npu\\ethosu_device.h" 1
# 48 "../../../../Library/StdDriver/src/npu\\ethosu_device.h"
struct NPU_REG;

struct ethosu_device
{
    volatile struct NPU_REG *reg;
    uint32_t secure;
    uint32_t privileged;
};
# 64 "../../../../Library/StdDriver/src/npu\\ethosu_device.h"
struct ethosu_device *ethosu_dev_init(const void *base_address, uint32_t secure_enable, uint32_t privilege_enable);




void ethosu_dev_deinit(struct ethosu_device *dev);




enum ethosu_error_codes ethosu_dev_axi_init(struct ethosu_device *dev);
# 87 "../../../../Library/StdDriver/src/npu\\ethosu_device.h"
void ethosu_dev_run_command_stream(struct ethosu_device *dev,
                                   const uint8_t *cmd_stream_ptr,
                                   uint32_t cms_length,
                                   const uint64_t *base_addr,
                                   int num_base_addr);




void ethosu_dev_print_err_status(struct ethosu_device *dev);





_Bool ethosu_dev_handle_interrupt(struct ethosu_device *dev);





void ethosu_dev_get_hw_info(struct ethosu_device *dev, struct ethosu_hw_info *hwinfo);





_Bool ethosu_dev_verify_access_state(struct ethosu_device *dev);





enum ethosu_error_codes ethosu_dev_soft_reset(struct ethosu_device *dev);







enum ethosu_error_codes ethosu_dev_set_clock_and_power(struct ethosu_device *dev,
                                                       enum ethosu_clock_q_request clock_q,
                                                       enum ethosu_power_q_request power_q);







_Bool ethosu_dev_verify_optimizer_config(struct ethosu_device *dev, uint32_t cfg_in, uint32_t id_in);
# 25 "../../../../Library/StdDriver/src/npu/ethosu_driver.c" 2
# 1 "../../../../Library/StdDriver/src/npu\\ethosu_log.h" 1
# 26 "../../../../Library/StdDriver/src/npu\\ethosu_log.h"
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 1 3
# 68 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
    typedef __builtin_va_list __va_list;
# 87 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
typedef struct __fpos_t_struct {
    unsigned long long int __pos;





    struct {
        unsigned int __state1, __state2;
    } __mbstate;
} fpos_t;
# 108 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
typedef struct __FILE FILE;
# 119 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
struct __FILE {
    union {
        long __FILE_alignment;



        char __FILE_size[84];

    } __FILE_opaque;
};
# 138 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern FILE __stdin, __stdout, __stderr;
extern FILE *__aeabi_stdin, *__aeabi_stdout, *__aeabi_stderr;
# 224 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int remove(const char * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int rename(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 243 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) FILE *tmpfile(void);






extern __attribute__((__nothrow__)) char *tmpnam(char * );
# 265 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fclose(FILE * ) __attribute__((__nonnull__(1)));
# 275 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fflush(FILE * );
# 285 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) FILE *fopen(const char * __restrict ,
                           const char * __restrict ) __attribute__((__nonnull__(1,2)));
# 329 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) FILE *freopen(const char * __restrict ,
                    const char * __restrict ,
                    FILE * __restrict ) __attribute__((__nonnull__(2,3)));
# 342 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) void setbuf(FILE * __restrict ,
                    char * __restrict ) __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) int setvbuf(FILE * __restrict ,
                   char * __restrict ,
                   int , size_t ) __attribute__((__nonnull__(1)));
# 370 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int fprintf(FILE * __restrict ,
                    const char * __restrict , ...) __attribute__((__nonnull__(1,2)));
# 393 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int _fprintf(FILE * __restrict ,
                     const char * __restrict , ...) __attribute__((__nonnull__(1,2)));





#pragma __printf_args
extern __attribute__((__nothrow__)) int printf(const char * __restrict , ...) __attribute__((__nonnull__(1)));






#pragma __printf_args
extern __attribute__((__nothrow__)) int _printf(const char * __restrict , ...) __attribute__((__nonnull__(1)));





#pragma __printf_args
extern __attribute__((__nothrow__)) int sprintf(char * __restrict , const char * __restrict , ...) __attribute__((__nonnull__(1,2)));








#pragma __printf_args
extern __attribute__((__nothrow__)) int _sprintf(char * __restrict , const char * __restrict , ...) __attribute__((__nonnull__(1,2)));






#pragma __printf_args
extern __attribute__((__nothrow__)) int __ARM_snprintf(char * __restrict , size_t ,
                     const char * __restrict , ...) __attribute__((__nonnull__(3)));


#pragma __printf_args
extern __attribute__((__nothrow__)) int snprintf(char * __restrict , size_t ,
                     const char * __restrict , ...) __attribute__((__nonnull__(3)));
# 460 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int _snprintf(char * __restrict , size_t ,
                      const char * __restrict , ...) __attribute__((__nonnull__(3)));





#pragma __scanf_args
extern __attribute__((__nothrow__)) int fscanf(FILE * __restrict ,
                    const char * __restrict , ...) __attribute__((__nonnull__(1,2)));
# 503 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __scanf_args
extern __attribute__((__nothrow__)) int _fscanf(FILE * __restrict ,
                     const char * __restrict , ...) __attribute__((__nonnull__(1,2)));





#pragma __scanf_args
extern __attribute__((__nothrow__)) int scanf(const char * __restrict , ...) __attribute__((__nonnull__(1)));








#pragma __scanf_args
extern __attribute__((__nothrow__)) int _scanf(const char * __restrict , ...) __attribute__((__nonnull__(1)));





#pragma __scanf_args
extern __attribute__((__nothrow__)) int sscanf(const char * __restrict ,
                    const char * __restrict , ...) __attribute__((__nonnull__(1,2)));
# 541 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __scanf_args
extern __attribute__((__nothrow__)) int _sscanf(const char * __restrict ,
                     const char * __restrict , ...) __attribute__((__nonnull__(1,2)));







extern __attribute__((__nothrow__)) int vfscanf(FILE * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) int vscanf(const char * __restrict , __va_list) __attribute__((__nonnull__(1)));
extern __attribute__((__nothrow__)) int vsscanf(const char * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));

extern __attribute__((__nothrow__)) int _vfscanf(FILE * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) int _vscanf(const char * __restrict , __va_list) __attribute__((__nonnull__(1)));
extern __attribute__((__nothrow__)) int _vsscanf(const char * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) int __ARM_vsscanf(const char * __restrict , const char * __restrict , __va_list) __attribute__((__nonnull__(1,2)));

extern __attribute__((__nothrow__)) int vprintf(const char * __restrict , __va_list ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int _vprintf(const char * __restrict , __va_list ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) int vfprintf(FILE * __restrict ,
                    const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));
# 584 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int vsprintf(char * __restrict ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));
# 594 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int __ARM_vsnprintf(char * __restrict , size_t ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(3)));

extern __attribute__((__nothrow__)) int vsnprintf(char * __restrict , size_t ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(3)));
# 609 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int _vsprintf(char * __restrict ,
                      const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));





extern __attribute__((__nothrow__)) int _vfprintf(FILE * __restrict ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));





extern __attribute__((__nothrow__)) int _vsnprintf(char * __restrict , size_t ,
                      const char * __restrict , __va_list ) __attribute__((__nonnull__(3)));
# 635 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int __ARM_asprintf(char ** , const char * __restrict , ...) __attribute__((__nonnull__(2)));
extern __attribute__((__nothrow__)) int __ARM_vasprintf(char ** , const char * __restrict , __va_list ) __attribute__((__nonnull__(2)));
# 649 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fgetc(FILE * ) __attribute__((__nonnull__(1)));
# 659 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) char *fgets(char * __restrict , int ,
                    FILE * __restrict ) __attribute__((__nonnull__(1,3)));
# 673 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fputc(int , FILE * ) __attribute__((__nonnull__(2)));
# 683 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fputs(const char * __restrict , FILE * __restrict ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) int getc(FILE * ) __attribute__((__nonnull__(1)));
# 704 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
    extern __attribute__((__nothrow__)) int (getchar)(void);
# 713 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) char *gets(char * ) __attribute__((__nonnull__(1)));
# 725 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int putc(int , FILE * ) __attribute__((__nonnull__(2)));
# 737 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
    extern __attribute__((__nothrow__)) int (putchar)(int );






extern __attribute__((__nothrow__)) int puts(const char * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int ungetc(int , FILE * ) __attribute__((__nonnull__(2)));
# 778 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) size_t fread(void * __restrict ,
                    size_t , size_t , FILE * __restrict ) __attribute__((__nonnull__(1,4)));
# 794 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) size_t __fread_bytes_avail(void * __restrict ,
                    size_t , FILE * __restrict ) __attribute__((__nonnull__(1,3)));
# 810 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) size_t fwrite(const void * __restrict ,
                    size_t , size_t , FILE * __restrict ) __attribute__((__nonnull__(1,4)));
# 822 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fgetpos(FILE * __restrict , fpos_t * __restrict ) __attribute__((__nonnull__(1,2)));
# 833 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fseek(FILE * , long int , int ) __attribute__((__nonnull__(1)));
# 850 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fsetpos(FILE * __restrict , const fpos_t * __restrict ) __attribute__((__nonnull__(1,2)));
# 863 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) long int ftell(FILE * ) __attribute__((__nonnull__(1)));
# 877 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) void rewind(FILE * ) __attribute__((__nonnull__(1)));
# 886 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) void clearerr(FILE * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int feof(FILE * ) __attribute__((__nonnull__(1)));




extern __attribute__((__nothrow__)) int ferror(FILE * ) __attribute__((__nonnull__(1)));




extern __attribute__((__nothrow__)) void perror(const char * );
# 917 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int _fisatty(FILE * ) __attribute__((__nonnull__(1)));



extern __attribute__((__nothrow__)) void __use_no_semihosting_swi(void);
extern __attribute__((__nothrow__)) void __use_no_semihosting(void);
# 27 "../../../../Library/StdDriver/src/npu\\ethosu_log.h" 2
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 1 3
# 58 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void *memcpy(void * __restrict ,
                    const void * __restrict , size_t ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) void *memmove(void * ,
                    const void * , size_t ) __attribute__((__nonnull__(1,2)));
# 77 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strcpy(char * __restrict , const char * __restrict ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) char *strncpy(char * __restrict , const char * __restrict , size_t ) __attribute__((__nonnull__(1,2)));
# 93 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strcat(char * __restrict , const char * __restrict ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) char *strncat(char * __restrict , const char * __restrict , size_t ) __attribute__((__nonnull__(1,2)));
# 117 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) int memcmp(const void * , const void * , size_t ) __attribute__((__nonnull__(1,2)));







extern __attribute__((__nothrow__)) int strcmp(const char * , const char * ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) int strncmp(const char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 141 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) int strcasecmp(const char * , const char * ) __attribute__((__nonnull__(1,2)));







extern __attribute__((__nothrow__)) int strncasecmp(const char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 158 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) int strcoll(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 169 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strxfrm(char * __restrict , const char * __restrict , size_t ) __attribute__((__nonnull__(2)));
# 193 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void *memchr(const void * , int , size_t ) __attribute__((__nonnull__(1)));
# 209 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strchr(const char * , int ) __attribute__((__nonnull__(1)));
# 218 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strcspn(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 232 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strpbrk(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 247 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strrchr(const char * , int ) __attribute__((__nonnull__(1)));
# 257 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strspn(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 270 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strstr(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 280 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strtok(char * __restrict , const char * __restrict ) __attribute__((__nonnull__(2)));
extern __attribute__((__nothrow__)) char *_strtok_r(char * , const char * , char ** ) __attribute__((__nonnull__(2,3)));
# 321 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void *memset(void * , int , size_t ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) char *strerror(int );







extern __attribute__((__nothrow__)) size_t strlen(const char * ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) size_t strnlen(const char * , size_t ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) size_t strlcpy(char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 369 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strlcat(char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 395 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void _membitcpybl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpybb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpyhl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpyhb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpywl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitcpywb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovebl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovebb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovehl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovehb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovewl(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
extern __attribute__((__nothrow__)) void _membitmovewb(void * , const void * , int , int , size_t ) __attribute__((__nonnull__(1,2)));
# 28 "../../../../Library/StdDriver/src/npu\\ethosu_log.h" 2
# 26 "../../../../Library/StdDriver/src/npu/ethosu_driver.c" 2


# 1 "../../../../Library/StdDriver/src/npu\\ethosu_config_u55.h" 1
# 29 "../../../../Library/StdDriver/src/npu/ethosu_driver.c" 2




# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\assert.h" 1 3
# 43 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\assert.h" 3
    extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void abort(void);
    extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void __aeabi_assert(const char *, const char *, int) __attribute__((__nonnull__(1,2)));
# 34 "../../../../Library/StdDriver/src/npu/ethosu_driver.c" 2
# 1 "../../../../Library/CMSIS/Core/Include\\cmsis_compiler.h" 1
# 47 "../../../../Library/CMSIS/Core/Include\\cmsis_compiler.h"
# 1 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 1
# 31 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
# 64 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpacked"

  struct __attribute__((packed)) T_UINT32 { uint32_t v; };
#pragma clang diagnostic pop



#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpacked"

  struct __attribute__((packed, aligned(1))) T_UINT16_WRITE { uint16_t v; };
#pragma clang diagnostic pop



#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpacked"

  struct __attribute__((packed, aligned(1))) T_UINT16_READ { uint16_t v; };
#pragma clang diagnostic pop



#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpacked"

  struct __attribute__((packed, aligned(1))) T_UINT32_WRITE { uint32_t v; };
#pragma clang diagnostic pop



#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpacked"

  struct __attribute__((packed, aligned(1))) T_UINT32_READ { uint32_t v; };
#pragma clang diagnostic pop
# 260 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __ROR(uint32_t op1, uint32_t op2)
{
  op2 %= 32U;
  if (op2 == 0U)
  {
    return op1;
  }
  return (op1 >> op2) | (op1 << (32U - op2));
}
# 295 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint8_t __CLZ(uint32_t value)
{
# 306 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
  if (value == 0U)
  {
    return 32U;
  }
  return __builtin_clz(value);
}
# 425 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __RRX(uint32_t value)
{
  uint32_t result;

  __asm volatile ("rrx %0, %1" : "=r" (result) : "r" (value) );
  return(result);
}
# 440 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint8_t __LDRBT(volatile uint8_t *ptr)
{
  uint32_t result;

  __asm volatile ("ldrbt %0, %1" : "=r" (result) : "Q" (*ptr) );
  return ((uint8_t) result);
}
# 455 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint16_t __LDRHT(volatile uint16_t *ptr)
{
  uint32_t result;

  __asm volatile ("ldrht %0, %1" : "=r" (result) : "Q" (*ptr) );
  return ((uint16_t) result);
}
# 470 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __LDRT(volatile uint32_t *ptr)
{
  uint32_t result;

  __asm volatile ("ldrt %0, %1" : "=r" (result) : "Q" (*ptr) );
  return(result);
}
# 485 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __STRBT(uint8_t value, volatile uint8_t *ptr)
{
  __asm volatile ("strbt %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) );
}
# 497 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __STRHT(uint16_t value, volatile uint16_t *ptr)
{
  __asm volatile ("strht %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) );
}
# 509 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __STRT(uint32_t value, volatile uint32_t *ptr)
{
  __asm volatile ("strt %1, %0" : "=Q" (*ptr) : "r" (value) );
}
# 584 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint8_t __LDAB(volatile uint8_t *ptr)
{
  uint32_t result;

  __asm volatile ("ldab %0, %1" : "=r" (result) : "Q" (*ptr) : "memory" );
  return ((uint8_t) result);
}
# 599 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint16_t __LDAH(volatile uint16_t *ptr)
{
  uint32_t result;

  __asm volatile ("ldah %0, %1" : "=r" (result) : "Q" (*ptr) : "memory" );
  return ((uint16_t) result);
}
# 614 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __LDA(volatile uint32_t *ptr)
{
  uint32_t result;

  __asm volatile ("lda %0, %1" : "=r" (result) : "Q" (*ptr) : "memory" );
  return(result);
}
# 629 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __STLB(uint8_t value, volatile uint8_t *ptr)
{
  __asm volatile ("stlb %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) : "memory" );
}
# 641 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __STLH(uint16_t value, volatile uint16_t *ptr)
{
  __asm volatile ("stlh %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) : "memory" );
}
# 653 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __STL(uint32_t value, volatile uint32_t *ptr)
{
  __asm volatile ("stl %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) : "memory" );
}
# 737 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __enable_irq(void)
{
  __asm volatile ("cpsie i" : : : "memory");
}
# 750 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __disable_irq(void)
{
  __asm volatile ("cpsid i" : : : "memory");
}
# 762 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __get_CONTROL(void)
{
  uint32_t result;

  __asm volatile ("MRS %0, control" : "=r" (result) );
  return(result);
}
# 792 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __set_CONTROL(uint32_t control)
{
  __asm volatile ("MSR control, %0" : : "r" (control) : "memory");
  __builtin_arm_isb(0xF);
}
# 818 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __get_IPSR(void)
{
  uint32_t result;

  __asm volatile ("MRS %0, ipsr" : "=r" (result) );
  return(result);
}







__attribute__((always_inline)) static __inline uint32_t __get_APSR(void)
{
  uint32_t result;

  __asm volatile ("MRS %0, apsr" : "=r" (result) );
  return(result);
}







__attribute__((always_inline)) static __inline uint32_t __get_xPSR(void)
{
  uint32_t result;

  __asm volatile ("MRS %0, xpsr" : "=r" (result) );
  return(result);
}







__attribute__((always_inline)) static __inline uint32_t __get_PSP(void)
{
  uint32_t result;

  __asm volatile ("MRS %0, psp" : "=r" (result) );
  return(result);
}
# 890 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __set_PSP(uint32_t topOfProcStack)
{
  __asm volatile ("MSR psp, %0" : : "r" (topOfProcStack) : );
}
# 914 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __get_MSP(void)
{
  uint32_t result;

  __asm volatile ("MRS %0, msp" : "=r" (result) );
  return(result);
}
# 944 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __set_MSP(uint32_t topOfMainStack)
{
  __asm volatile ("MSR msp, %0" : : "r" (topOfMainStack) : );
}
# 995 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __get_PRIMASK(void)
{
  uint32_t result;

  __asm volatile ("MRS %0, primask" : "=r" (result) );
  return(result);
}
# 1025 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __set_PRIMASK(uint32_t priMask)
{
  __asm volatile ("MSR primask, %0" : : "r" (priMask) : "memory");
}
# 1053 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __enable_fault_irq(void)
{
  __asm volatile ("cpsie f" : : : "memory");
}







__attribute__((always_inline)) static __inline void __disable_fault_irq(void)
{
  __asm volatile ("cpsid f" : : : "memory");
}







__attribute__((always_inline)) static __inline uint32_t __get_BASEPRI(void)
{
  uint32_t result;

  __asm volatile ("MRS %0, basepri" : "=r" (result) );
  return(result);
}
# 1105 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __set_BASEPRI(uint32_t basePri)
{
  __asm volatile ("MSR basepri, %0" : : "r" (basePri) : "memory");
}
# 1130 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __set_BASEPRI_MAX(uint32_t basePri)
{
  __asm volatile ("MSR basepri_max, %0" : : "r" (basePri) : "memory");
}







__attribute__((always_inline)) static __inline uint32_t __get_FAULTMASK(void)
{
  uint32_t result;

  __asm volatile ("MRS %0, faultmask" : "=r" (result) );
  return(result);
}
# 1171 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __set_FAULTMASK(uint32_t faultMask)
{
  __asm volatile ("MSR faultmask, %0" : : "r" (faultMask) : "memory");
}
# 1208 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __get_PSPLIM(void)
{






  uint32_t result;
  __asm volatile ("MRS %0, psplim" : "=r" (result) );
  return result;

}
# 1256 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __set_PSPLIM(uint32_t ProcStackPtrLimit)
{






  __asm volatile ("MSR psplim, %0" : : "r" (ProcStackPtrLimit));

}
# 1300 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline uint32_t __get_MSPLIM(void)
{






  uint32_t result;
  __asm volatile ("MRS %0, msplim" : "=r" (result) );
  return result;

}
# 1347 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline void __set_MSPLIM(uint32_t MainStackPtrLimit)
{






  __asm volatile ("MSR msplim, %0" : : "r" (MainStackPtrLimit));

}
# 1491 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 3
__attribute__((always_inline)) static __inline int32_t __SMMLA (int32_t op1, int32_t op2, int32_t op3)
{
  int32_t result;

  __asm volatile ("smmla %0, %1, %2, %3" : "=r" (result): "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}
# 48 "../../../../Library/CMSIS/Core/Include\\cmsis_compiler.h" 2
# 35 "../../../../Library/StdDriver/src/npu/ethosu_driver.c" 2
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\inttypes.h" 1 3
# 218 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\inttypes.h" 3
typedef struct imaxdiv_t { intmax_t quot, rem; } imaxdiv_t;






__attribute__((__nothrow__)) intmax_t strtoimax(const char * __restrict ,
                   char ** __restrict , int ) __attribute__((__nonnull__(1)));

__attribute__((__nothrow__)) uintmax_t strtoumax(const char * __restrict ,
                    char ** __restrict , int ) __attribute__((__nonnull__(1)));


__attribute__((__nothrow__)) intmax_t wcstoimax(const wchar_t * __restrict ,
                   wchar_t ** __restrict , int ) __attribute__((__nonnull__(1)));
__attribute__((__nothrow__)) uintmax_t wcstoumax(const wchar_t * __restrict ,
                    wchar_t ** __restrict , int ) __attribute__((__nonnull__(1)));

extern __attribute__((__nothrow__)) __attribute__((__const__)) intmax_t imaxabs(intmax_t );





extern __attribute__((__nothrow__)) __attribute__((__const__)) imaxdiv_t imaxdiv(intmax_t , intmax_t );
# 36 "../../../../Library/StdDriver/src/npu/ethosu_driver.c" 2



# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 1 3
# 96 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
typedef struct div_t { int quot, rem; } div_t;

typedef struct ldiv_t { long int quot, rem; } ldiv_t;


typedef struct lldiv_t { long long quot, rem; } lldiv_t;
# 139 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int __aeabi_MB_CUR_MAX(void);
# 158 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) double atof(const char * ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) int atoi(const char * ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) long int atol(const char * ) __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) long long atoll(const char * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) double strtod(const char * __restrict , char ** __restrict ) __attribute__((__nonnull__(1)));
# 206 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) float strtof(const char * __restrict , char ** __restrict ) __attribute__((__nonnull__(1)));
extern __attribute__((__nothrow__)) long double strtold(const char * __restrict , char ** __restrict ) __attribute__((__nonnull__(1)));




extern __attribute__((__nothrow__)) long int strtol(const char * __restrict ,
                        char ** __restrict , int ) __attribute__((__nonnull__(1)));
# 243 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) unsigned long int strtoul(const char * __restrict ,
                                       char ** __restrict , int ) __attribute__((__nonnull__(1)));
# 275 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) long long strtoll(const char * __restrict ,
                                  char ** __restrict , int )
                          __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) unsigned long long strtoull(const char * __restrict ,
                                            char ** __restrict , int )
                                   __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) int rand(void);
# 303 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) void srand(unsigned int );
# 313 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
struct _rand_state { int __x[57]; };
extern __attribute__((__nothrow__)) int _rand_r(struct _rand_state *);
extern __attribute__((__nothrow__)) void _srand_r(struct _rand_state *, unsigned int);
struct _ANSI_rand_state { int __x[1]; };
extern __attribute__((__nothrow__)) int _ANSI_rand_r(struct _ANSI_rand_state *);
extern __attribute__((__nothrow__)) void _ANSI_srand_r(struct _ANSI_rand_state *, unsigned int);





extern __attribute__((__nothrow__)) void *calloc(size_t , size_t );





extern __attribute__((__nothrow__)) void free(void * );







extern __attribute__((__nothrow__)) void *malloc(size_t );





extern __attribute__((__nothrow__)) void *realloc(void * , size_t );
# 374 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
typedef int (*__heapprt)(void *, char const *, ...);
extern __attribute__((__nothrow__)) void __heapstats(int (* )(void * ,
                                           char const * , ...),
                        void * ) __attribute__((__nonnull__(1)));
# 390 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int __heapvalid(int (* )(void * ,
                                           char const * , ...),
                       void * , int ) __attribute__((__nonnull__(1)));
# 411 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void abort(void);
# 422 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int atexit(void (* )(void)) __attribute__((__nonnull__(1)));
# 444 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void exit(int );
# 460 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__noreturn__)) void _Exit(int );
# 471 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) char *getenv(const char * ) __attribute__((__nonnull__(1)));
# 484 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int system(const char * );
# 497 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern void *bsearch(const void * , const void * ,
              size_t , size_t ,
              int (* )(const void *, const void *)) __attribute__((__nonnull__(1,2,5)));
# 532 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern void qsort(void * , size_t , size_t ,
           int (* )(const void *, const void *)) __attribute__((__nonnull__(1,4)));
# 560 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__const__)) int abs(int );






extern __attribute__((__nothrow__)) __attribute__((__const__)) div_t div(int , int );
# 579 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__const__)) long int labs(long int );
# 589 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__const__)) ldiv_t ldiv(long int , long int );
# 610 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__const__)) long long llabs(long long );
# 620 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) __attribute__((__const__)) lldiv_t lldiv(long long , long long );
# 644 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
typedef struct __sdiv32by16 { long quot, rem; } __sdiv32by16;
typedef struct __udiv32by16 { unsigned long quot, rem; } __udiv32by16;

typedef struct __sdiv64by32 { long rem, quot; } __sdiv64by32;

__attribute__((__value_in_regs__)) extern __attribute__((__nothrow__)) __attribute__((__const__)) __sdiv32by16 __rt_sdiv32by16(
     int ,
     short int );



__attribute__((__value_in_regs__)) extern __attribute__((__nothrow__)) __attribute__((__const__)) __udiv32by16 __rt_udiv32by16(
     unsigned int ,
     unsigned short );



__attribute__((__value_in_regs__)) extern __attribute__((__nothrow__)) __attribute__((__const__)) __sdiv64by32 __rt_sdiv64by32(
     int , unsigned int ,
     int );







extern __attribute__((__nothrow__)) unsigned int __fp_status(unsigned int , unsigned int );
# 705 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int mblen(const char * , size_t );
# 720 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int mbtowc(wchar_t * __restrict ,
                   const char * __restrict , size_t );
# 739 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) int wctomb(char * , wchar_t );
# 761 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) size_t mbstowcs(wchar_t * __restrict ,
                      const char * __restrict , size_t ) __attribute__((__nonnull__(2)));
# 779 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) size_t wcstombs(char * __restrict ,
                      const wchar_t * __restrict , size_t ) __attribute__((__nonnull__(2)));
# 798 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
extern __attribute__((__nothrow__)) void __use_realtime_heap(void);
extern __attribute__((__nothrow__)) void __use_realtime_division(void);
extern __attribute__((__nothrow__)) void __use_two_region_memory(void);
extern __attribute__((__nothrow__)) void __use_no_heap(void);
extern __attribute__((__nothrow__)) void __use_no_heap_region(void);

extern __attribute__((__nothrow__)) char const *__C_library_version_string(void);
extern __attribute__((__nothrow__)) int __C_library_version_number(void);
# 40 "../../../../Library/StdDriver/src/npu/ethosu_driver.c" 2
# 60 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
enum DRIVER_ACTION_e
{
    RESERVED = 0,
    OPTIMIZER_CONFIG = 1,
    COMMAND_STREAM = 2,
    NOP = 5,
};


struct cop_data_s
{
    union
    {

        struct
        {
            uint8_t driver_action_command;
            uint8_t reserved;


            union
            {

                struct
                {
                    uint16_t rel_nbr : 4;
                    uint16_t patch_nbr : 4;
                    uint16_t opt_cfg_reserved : 8;
                };


                struct
                {
                    uint16_t length;
                };

                uint16_t driver_action_data;
            };
        };

        uint32_t word;
    };
};


struct opt_cfg_s
{
    struct cop_data_s da_data;
    uint32_t cfg;
    uint32_t id;
};






static struct ethosu_driver *registered_drivers = 0;
# 129 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
void __attribute__((weak)) ethosu_flush_dcache(uint32_t *p, size_t bytes)
{
    ((void)p);
    ((void)bytes);
}





void __attribute__((weak)) ethosu_invalidate_dcache(uint32_t *p, size_t bytes)
{
    ((void)p);
    ((void)bytes);
}
# 153 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
struct ethosu_semaphore_t
{
    uint8_t count;
};

static void *ethosu_mutex;
static void *ethosu_semaphore;

void *__attribute__((weak)) ethosu_mutex_create(void)
{
    return 0;
}

void __attribute__((weak)) ethosu_mutex_destroy(void *mutex)
{
    ((void)mutex);
}

int __attribute__((weak)) ethosu_mutex_lock(void *mutex)
{
    ((void)mutex);
    return 0;
}

int __attribute__((weak)) ethosu_mutex_unlock(void *mutex)
{
    ((void)mutex);
    return 0;
}


void *__attribute__((weak)) ethosu_semaphore_create(void)
{
    struct ethosu_semaphore_t *sem = malloc(sizeof(*sem));
    sem->count = 0;
    return sem;
}

void __attribute__((weak)) ethosu_semaphore_destroy(void *sem)
{
    free((struct ethosu_semaphore_t *)sem);
}




int __attribute__((weak)) ethosu_semaphore_take(void *sem)
{
    struct ethosu_semaphore_t *s = sem;

    while (s->count == 0)
    {



        __builtin_arm_wfe();


    }

    s->count = 0;
    return 0;
}


int __attribute__((weak)) ethosu_semaphore_give(void *sem)
{
    struct ethosu_semaphore_t *s = sem;
    s->count = 1;

    __builtin_arm_sev();

    return 0;
}





void __attribute__((weak)) ethosu_inference_begin(struct ethosu_driver *drv, void *user_arg)
{
    ((void)user_arg);
    ((void)drv);
}

void __attribute__((weak)) ethosu_inference_end(struct ethosu_driver *drv, void *user_arg)
{
    ((void)user_arg);
    ((void)drv);
}




static void ethosu_register_driver(struct ethosu_driver *drv)
{

    drv->next = registered_drivers;
    registered_drivers = drv;

                                                                                       ;
}

static int ethosu_deregister_driver(struct ethosu_driver *drv)
{
    struct ethosu_driver *cur = registered_drivers;
    struct ethosu_driver **prev = &registered_drivers;

    while (cur != 0)
    {
        if (cur == drv)
        {
            *prev = cur->next;
                                                               ;
            return 0;
        }

        prev = &cur->next;
        cur = cur->next;
    }

    fprintf((& __stderr), "E: " "No NPU driver handle registered at address %p." " (%s:%d)\n", drv, strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 274);

    return -1;
}

static struct ethosu_driver *ethosu_find_and_reserve_driver(void)
{
    struct ethosu_driver *drv = registered_drivers;

    while (drv != 0)
    {
        if (!drv->reserved)
        {
            drv->reserved = 1;
                                                            ;
            return drv;
        }

        drv = drv->next;
    }

    fprintf((& __stdout), "W: " "No NPU driver handle available." "\n");

    return 0;
}

static void ethosu_reset_job(struct ethosu_driver *drv)
{
    memset(&drv->job, 0, sizeof(struct ethosu_job));
}

static int handle_optimizer_config(struct ethosu_driver *drv, struct opt_cfg_s *opt_cfg_p)
{
                                                                                                             ;

    if (ethosu_dev_verify_optimizer_config(drv->dev, opt_cfg_p->cfg, opt_cfg_p->id) != 1)
    {
        return -1;
    }

    return 0;
}

static int handle_command_stream(struct ethosu_driver *drv, const uint8_t *cmd_stream, const int cms_length)
{
    uint32_t cms_bytes = cms_length * 4;
    ptrdiff_t cmd_stream_ptr = (ptrdiff_t)cmd_stream;

                                                                                           ;

    if (0 != ((ptrdiff_t)cmd_stream & (0xF)))
    {
        fprintf((& __stderr), "E: " "Command stream addr %p not aligned to 16 bytes" " (%s:%d)\n", cmd_stream, strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 326);
        return -1;
    }


    for (int i = 0; i < drv->job.num_base_addr; i++)
    {
        if (0 != (drv->job.base_addr[i] & (0xF)))
        {
            fprintf((& __stderr), "E: " "Base addr %d: 0x%llx not aligned to 16 bytes" " (%s:%d)\n", i, drv->job.base_addr[i], strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 335);
            return -1;
        }
    }






    if (drv->job.base_addr_size != 0)
    {
        ethosu_flush_dcache((uint32_t *)cmd_stream_ptr, cms_bytes);

        for (int i = 0; i < drv->job.num_base_addr; i++)
        {
            ethosu_flush_dcache((uint32_t *)(uintptr_t)drv->job.base_addr[i], drv->job.base_addr_size[i]);
        }
    }
    else
    {
        ethosu_flush_dcache(0, 0);
    }


    if (ethosu_request_power(drv))
    {
        fprintf((& __stderr), "E: " "Failed to request power" " (%s:%d)\n", strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 362);
        return -1;
    }

    drv->job.state = ETHOSU_JOB_RUNNING;


    ethosu_inference_begin(drv, drv->job.user_arg);


    ethosu_dev_run_command_stream(drv->dev, cmd_stream, cms_bytes, drv->job.base_addr, drv->job.num_base_addr);

    return 0;
}




void __attribute__((weak)) ethosu_irq_handler(struct ethosu_driver *drv)
{
                                           ;

    drv->job.state = ETHOSU_JOB_DONE;

    if (!ethosu_dev_handle_interrupt(drv->dev))
    {
        drv->status_error = 1;
    }


    ethosu_semaphore_give(drv->semaphore);
}





int ethosu_init(struct ethosu_driver *drv,
                const void *base_address,
                const void *fast_memory,
                const size_t fast_memory_size,
                uint32_t secure_enable,
                uint32_t privilege_enable)
{






                              ;

    if (!ethosu_mutex)
    {
        ethosu_mutex = ethosu_mutex_create();
    }

    if (!ethosu_semaphore)
    {
        ethosu_semaphore = ethosu_semaphore_create();
    }

    drv->fast_memory = (uint32_t)fast_memory;
    drv->fast_memory_size = fast_memory_size;
    drv->power_request_counter = 0;


    drv->dev = ethosu_dev_init(base_address, secure_enable, privilege_enable);

    if (drv->dev == 0)
    {
        fprintf((& __stderr), "E: " "Failed to initialize Ethos-U device" " (%s:%d)\n", strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 433);
        return -1;
    }

    drv->semaphore = ethosu_semaphore_create();
    drv->status_error = 0;

    ethosu_reset_job(drv);

    ethosu_register_driver(drv);

    return 0;
}

void ethosu_deinit(struct ethosu_driver *drv)
{
    ethosu_deregister_driver(drv);
    ethosu_semaphore_destroy(drv->semaphore);
    ethosu_dev_deinit(drv->dev);
    drv->dev = 0;
}

int ethosu_soft_reset(struct ethosu_driver *drv)
{

    if (ethosu_dev_soft_reset(drv->dev) != ETHOSU_SUCCESS)
    {
        fprintf((& __stderr), "E: " "Failed to soft-reset NPU" " (%s:%d)\n", strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 460);
        return -1;
    }


    ethosu_dev_set_clock_and_power(drv->dev,
                                   drv->power_request_counter > 0 ? ETHOSU_CLOCK_Q_DISABLE : ETHOSU_CLOCK_Q_ENABLE,
                                   drv->power_request_counter > 0 ? ETHOSU_POWER_Q_DISABLE : ETHOSU_POWER_Q_ENABLE);

    return 0;
}

int ethosu_request_power(struct ethosu_driver *drv)
{

    if (drv->power_request_counter++ == 0)
    {


        if (ethosu_soft_reset(drv))
        {
            fprintf((& __stderr), "E: " "Failed to request power for Ethos-U" " (%s:%d)\n", strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 481);
            drv->power_request_counter--;
            return -1;
        }
    }

    return 0;
}

void ethosu_release_power(struct ethosu_driver *drv)
{
    if (drv->power_request_counter == 0)
    {
        fprintf((& __stdout), "W: " "No power request left to release, reference counter is 0" "\n");
    }
    else
    {

        if (--drv->power_request_counter == 0)
        {
            ethosu_dev_set_clock_and_power(drv->dev, ETHOSU_CLOCK_Q_ENABLE, ETHOSU_POWER_Q_ENABLE);
        }
    }
}

void ethosu_get_driver_version(struct ethosu_driver_version *ver)
{
    ((ver != 0) ? (void)0 : __aeabi_assert("ver != NULL", "../../../../Library/StdDriver/src/npu/ethosu_driver.c", 508),
# 508 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
#pragma clang diagnostic push
# 508 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
#pragma clang diagnostic ignored "-Wassume"
# 508 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
    (__builtin_assume)((ver != 0)?1:0)
# 508 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
#pragma clang diagnostic pop
# 508 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
    );
    ver->major = 0;
    ver->minor = 16;
    ver->patch = 0;
}

void ethosu_get_hw_info(struct ethosu_driver *drv, struct ethosu_hw_info *hw)
{
    ((hw != 0) ? (void)0 : __aeabi_assert("hw != NULL", "../../../../Library/StdDriver/src/npu/ethosu_driver.c", 516),
# 516 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
#pragma clang diagnostic push
# 516 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
#pragma clang diagnostic ignored "-Wassume"
# 516 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
    (__builtin_assume)((hw != 0)?1:0)
# 516 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
#pragma clang diagnostic pop
# 516 "../../../../Library/StdDriver/src/npu/ethosu_driver.c"
    );
    ethosu_dev_get_hw_info(drv->dev, hw);
}

int ethosu_wait(struct ethosu_driver *drv, _Bool block)
{
    int ret = 0;

    switch (drv->job.state)
    {
        case ETHOSU_JOB_IDLE:
            fprintf((& __stderr), "E: " "Inference job not running..." " (%s:%d)\n", strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 527);
            ret = -2;
            break;

        case ETHOSU_JOB_RUNNING:
            if (!block)
            {

                ret = 1;
                break;
            }


        case ETHOSU_JOB_DONE:



            ethosu_semaphore_take(drv->semaphore);


            ethosu_inference_end(drv, drv->job.user_arg);


            ethosu_release_power(drv);


            if (drv->status_error)
            {
                fprintf((& __stderr), "E: " "NPU error(s) occured during inference." " (%s:%d)\n", strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 555);
                ethosu_dev_print_err_status(drv->dev);


                (void)ethosu_soft_reset(drv);

                drv->status_error = 0;

                ret = -1;
            }

            if (ret == 0)
            {

                if (drv->job.base_addr_size != 0)
                {
                    for (int i = 0; i < drv->job.num_base_addr; i++)
                    {
                        ethosu_invalidate_dcache((uint32_t *)(uintptr_t)drv->job.base_addr[i], drv->job.base_addr_size[i]);
                    }
                }
                else
                {
                    ethosu_invalidate_dcache(0, 0);
                }

                                                               ;
            }


            ethosu_reset_job(drv);
            break;

        default:
            fprintf((& __stderr), "E: " "Unexpected job state" " (%s:%d)\n", strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 589);
            ethosu_reset_job(drv);
            ret = -1;
            break;
    }


    return ret;
}

int ethosu_invoke_async(struct ethosu_driver *drv,
                        const void *custom_data_ptr,
                        const int custom_data_size,
                        const uint64_t *base_addr,
                        const size_t *base_addr_size,
                        const int num_base_addr,
                        void *user_arg)
{

    const struct cop_data_s *data_ptr = custom_data_ptr;
    const struct cop_data_s *data_end = (struct cop_data_s *)((ptrdiff_t)custom_data_ptr + custom_data_size);


    if (drv->job.state != ETHOSU_JOB_IDLE)
    {
        fprintf((& __stderr), "E: " "Inference already running, or waiting to be cleared..." " (%s:%d)\n", strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 614);
        return -1;
    }

    drv->job.state = ETHOSU_JOB_IDLE;
    drv->job.custom_data_ptr = custom_data_ptr;
    drv->job.custom_data_size = custom_data_size;
    drv->job.base_addr = base_addr;
    drv->job.base_addr_size = base_addr_size;
    drv->job.num_base_addr = num_base_addr;
    drv->job.user_arg = user_arg;


    if (data_ptr->word != ('1' << 24 | 'P' << 16 | 'O' << 8 | 'C'))
    {
        fprintf((& __stderr), "E: " "Custom Operator Payload: %" "u" " is not correct, expected %x" " (%s:%d)\n", data_ptr->word, ('1' << 24 | 'P' << 16 | 'O' << 8 | 'C'), strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 629);
        goto err;
    }


    if ((custom_data_size % 4) != 0)
    {
        fprintf((& __stderr), "E: " "custom_data_size=0x%x not a multiple of 4" " (%s:%d)\n", custom_data_size, strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 636);
        goto err;
    }

    data_ptr++;


    if (drv->fast_memory != 0 && num_base_addr >= 2)
    {
        uint64_t *fast_memory = (uint64_t *)&base_addr[2];

        if (base_addr_size != 0 && base_addr_size[2] > drv->fast_memory_size)
        {
            fprintf((& __stderr), "E: " "Fast memory area too small. fast_memory_size=%u, base_addr_size=%u" " (%s:%d)\n", drv->fast_memory_size, base_addr_size[2], strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 651);


            goto err;
        }

        *fast_memory = drv->fast_memory;
    }

    drv->status_error = 0;

    struct opt_cfg_s *opt_cfg_p;
    void *command_stream;
    int cms_length;


    while (data_ptr < data_end)
    {
        switch (data_ptr->driver_action_command)
        {
            case OPTIMIZER_CONFIG:
                                             ;
                opt_cfg_p = (struct opt_cfg_s *)data_ptr;

                if (handle_optimizer_config(drv, opt_cfg_p) < 0)
                {
                    goto err;
                }

                data_ptr += 1 + 2;
                break;

            case COMMAND_STREAM:

                                           ;
                command_stream = (uint8_t *)(data_ptr) + sizeof(struct cop_data_s);
                cms_length = (data_ptr->reserved << 16) | data_ptr->length;

                if (handle_command_stream(drv, command_stream, cms_length) < 0)
                {
                    goto err;
                }

                data_ptr += 1 + cms_length;
                break;

            case NOP:
                                ;
                data_ptr += 1;
                break;

            default:
                fprintf((& __stderr), "E: " "UNSUPPORTED driver_action_command: %d" " (%s:%d)\n", data_ptr->driver_action_command, strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 701);
                goto err;
                break;
        }
    }

    return 0;
err:
    fprintf((& __stderr), "E: " "Failed to invoke inference." " (%s:%d)\n", strrchr("/" "../../../../Library/StdDriver/src/npu/ethosu_driver.c", '/') + 1, 709);
    ethosu_reset_job(drv);
    return -1;
}

int ethosu_invoke_v3(struct ethosu_driver *drv,
                     const void *custom_data_ptr,
                     const int custom_data_size,
                     const uint64_t *base_addr,
                     const size_t *base_addr_size,
                     const int num_base_addr,
                     void *user_arg)
{
    if (ethosu_invoke_async(
                drv, custom_data_ptr, custom_data_size, base_addr, base_addr_size, num_base_addr, user_arg) < 0)
    {
        return -1;
    }

    return ethosu_wait(drv, 1);
}

struct ethosu_driver *ethosu_reserve_driver(void)
{
    struct ethosu_driver *drv = 0;

    do
    {

        ethosu_mutex_lock(ethosu_mutex);
        drv = ethosu_find_and_reserve_driver();

        ethosu_mutex_unlock(ethosu_mutex);

        if (drv != 0)
        {
            break;
        }

                                                                        ;

        ethosu_semaphore_take(ethosu_semaphore);

    } while (1);

    return drv;
}

void ethosu_release_driver(struct ethosu_driver *drv)
{

    ethosu_mutex_lock(ethosu_mutex);

    if (drv != 0 && drv->reserved)
    {
        if (drv->job.state == ETHOSU_JOB_RUNNING || drv->job.state == ETHOSU_JOB_DONE)
        {

            if (ethosu_wait(drv, 0) == 1)
            {

                drv->power_request_counter = 0;
                ethosu_soft_reset(drv);
                ethosu_reset_job(drv);
                drv->status_error = 0;

                ethosu_semaphore_give(drv->semaphore);
            }
        }

        drv->reserved = 0;
                                                       ;

        ethosu_semaphore_give(ethosu_semaphore);
    }


    ethosu_mutex_unlock(ethosu_mutex);
}
