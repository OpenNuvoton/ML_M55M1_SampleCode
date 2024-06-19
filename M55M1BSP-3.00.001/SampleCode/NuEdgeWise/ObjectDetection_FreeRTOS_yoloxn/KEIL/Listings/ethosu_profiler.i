# 1 "../NPU/ethosu_profiler.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 391 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "../NPU/ethosu_profiler.c" 2
# 18 "../NPU/ethosu_profiler.c"
# 1 "../NPU/include\\ethosu_profiler.h" 1
# 20 "../NPU/include\\ethosu_profiler.h"
# 1 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h" 1
# 26 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdint.h" 1 3
# 56 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdint.h" 3
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
# 27 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h" 2

# 1 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h" 1
# 26 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
# 1 "../../../../Library/StdDriver/inc/npu/ethosu_types.h" 1
# 32 "../../../../Library/StdDriver/inc/npu/ethosu_types.h"
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
# 27 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h" 2

# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdbool.h" 1 3
# 29 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h" 2
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stddef.h" 1 3
# 38 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stddef.h" 3
  typedef signed int ptrdiff_t;
# 53 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stddef.h" 3
    typedef unsigned int size_t;
# 71 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stddef.h" 3
      typedef unsigned short wchar_t;
# 95 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stddef.h" 3
  typedef long double max_align_t;
# 30 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h" 2
# 49 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
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
# 104 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
void ethosu_irq_handler(struct ethosu_driver *drv);
# 115 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
void ethosu_flush_dcache(uint32_t *p, size_t bytes);
# 126 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
void ethosu_invalidate_dcache(uint32_t *p, size_t bytes);







void *ethosu_mutex_create(void);







void *ethosu_semaphore_create(void);







int ethosu_mutex_lock(void *mutex);







int ethosu_mutex_unlock(void *mutex);







int ethosu_semaphore_take(void *sem);







int ethosu_semaphore_give(void *sem);







void ethosu_inference_begin(struct ethosu_driver *drv, void *user_arg);







void ethosu_inference_end(struct ethosu_driver *drv, void *user_arg);
# 200 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
uint64_t ethosu_address_remap(uint64_t address, int index);
# 217 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
int ethosu_init(struct ethosu_driver *drv,
                const void *base_address,
                const void *fast_memory,
                const size_t fast_memory_size,
                uint32_t secure_enable,
                uint32_t privilege_enable);






void ethosu_deinit(struct ethosu_driver *drv);







int ethosu_soft_reset(struct ethosu_driver *drv);
# 247 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
int ethosu_request_power(struct ethosu_driver *drv);







void ethosu_release_power(struct ethosu_driver *drv);






void ethosu_get_driver_version(struct ethosu_driver_version *ver);







void ethosu_get_hw_info(struct ethosu_driver *drv, struct ethosu_hw_info *hw);
# 285 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
int ethosu_invoke_v3(struct ethosu_driver *drv,
                     const void *custom_data_ptr,
                     const int custom_data_size,
                     const uint64_t *base_addr,
                     const size_t *base_addr_size,
                     const int num_base_addr,
                     void *user_arg);
# 302 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
int ethosu_invoke_async(struct ethosu_driver *drv,
                        const void *custom_data_ptr,
                        const int custom_data_size,
                        const uint64_t *base_addr,
                        const size_t *base_addr_size,
                        const int num_base_addr,
                        void *user_arg);
# 319 "../../../../Library/StdDriver/inc/npu/ethosu_driver.h"
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
# 29 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h" 2
# 55 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
enum ethosu_pmu_event_type
{
    ETHOSU_PMU_NO_EVENT = 0,
    ETHOSU_PMU_CYCLE,
    ETHOSU_PMU_NPU_IDLE,
    ETHOSU_PMU_CC_STALLED_ON_BLOCKDEP,
    ETHOSU_PMU_CC_STALLED_ON_SHRAM_RECONFIG,
    ETHOSU_PMU_NPU_ACTIVE,
    ETHOSU_PMU_MAC_ACTIVE,
    ETHOSU_PMU_MAC_ACTIVE_8BIT,
    ETHOSU_PMU_MAC_ACTIVE_16BIT,
    ETHOSU_PMU_MAC_DPU_ACTIVE,
    ETHOSU_PMU_MAC_STALLED_BY_WD_ACC,
    ETHOSU_PMU_MAC_STALLED_BY_WD,
    ETHOSU_PMU_MAC_STALLED_BY_ACC,
    ETHOSU_PMU_MAC_STALLED_BY_IB,
    ETHOSU_PMU_MAC_ACTIVE_32BIT,
    ETHOSU_PMU_MAC_STALLED_BY_INT_W,
    ETHOSU_PMU_MAC_STALLED_BY_INT_ACC,
    ETHOSU_PMU_AO_ACTIVE,
    ETHOSU_PMU_AO_ACTIVE_8BIT,
    ETHOSU_PMU_AO_ACTIVE_16BIT,
    ETHOSU_PMU_AO_STALLED_BY_OFMP_OB,
    ETHOSU_PMU_AO_STALLED_BY_OFMP,
    ETHOSU_PMU_AO_STALLED_BY_OB,
    ETHOSU_PMU_AO_STALLED_BY_ACC_IB,
    ETHOSU_PMU_AO_STALLED_BY_ACC,
    ETHOSU_PMU_AO_STALLED_BY_IB,
    ETHOSU_PMU_WD_ACTIVE,
    ETHOSU_PMU_WD_STALLED,
    ETHOSU_PMU_WD_STALLED_BY_WS,
    ETHOSU_PMU_WD_STALLED_BY_WD_BUF,
    ETHOSU_PMU_WD_PARSE_ACTIVE,
    ETHOSU_PMU_WD_PARSE_STALLED,
    ETHOSU_PMU_WD_PARSE_STALLED_IN,
    ETHOSU_PMU_WD_PARSE_STALLED_OUT,
    ETHOSU_PMU_WD_TRANS_WS,
    ETHOSU_PMU_WD_TRANS_WB,
    ETHOSU_PMU_WD_TRANS_DW0,
    ETHOSU_PMU_WD_TRANS_DW1,
    ETHOSU_PMU_AXI0_RD_TRANS_ACCEPTED,
    ETHOSU_PMU_AXI0_RD_TRANS_COMPLETED,
    ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED,
    ETHOSU_PMU_AXI0_RD_TRAN_REQ_STALLED,
    ETHOSU_PMU_AXI0_WR_TRANS_ACCEPTED,
    ETHOSU_PMU_AXI0_WR_TRANS_COMPLETED_M,
    ETHOSU_PMU_AXI0_WR_TRANS_COMPLETED_S,
    ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN,
    ETHOSU_PMU_AXI0_WR_TRAN_REQ_STALLED,
    ETHOSU_PMU_AXI0_WR_DATA_BEAT_STALLED,
    ETHOSU_PMU_AXI0_ENABLED_CYCLES,
    ETHOSU_PMU_AXI0_RD_STALL_LIMIT,
    ETHOSU_PMU_AXI0_WR_STALL_LIMIT,
    ETHOSU_PMU_AXI_LATENCY_ANY,
    ETHOSU_PMU_AXI_LATENCY_32,
    ETHOSU_PMU_AXI_LATENCY_64,
    ETHOSU_PMU_AXI_LATENCY_128,
    ETHOSU_PMU_AXI_LATENCY_256,
    ETHOSU_PMU_AXI_LATENCY_512,
    ETHOSU_PMU_AXI_LATENCY_1024,
    ETHOSU_PMU_ECC_DMA,
    ETHOSU_PMU_ECC_SB0,
    ETHOSU_PMU_AXI1_RD_TRANS_ACCEPTED,
    ETHOSU_PMU_AXI1_RD_TRANS_COMPLETED,
    ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED,
    ETHOSU_PMU_AXI1_RD_TRAN_REQ_STALLED,
    ETHOSU_PMU_AXI1_WR_TRANS_ACCEPTED,
    ETHOSU_PMU_AXI1_WR_TRANS_COMPLETED_M,
    ETHOSU_PMU_AXI1_WR_TRANS_COMPLETED_S,
    ETHOSU_PMU_AXI1_WR_DATA_BEAT_WRITTEN,
    ETHOSU_PMU_AXI1_WR_TRAN_REQ_STALLED,
    ETHOSU_PMU_AXI1_WR_DATA_BEAT_STALLED,
    ETHOSU_PMU_AXI1_ENABLED_CYCLES,
    ETHOSU_PMU_AXI1_RD_STALL_LIMIT,
    ETHOSU_PMU_AXI1_WR_STALL_LIMIT,
    ETHOSU_PMU_ECC_SB1,

    ETHOSU_PMU_SENTINEL
};
# 142 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
void ETHOSU_PMU_Enable(struct ethosu_driver *drv);




void ETHOSU_PMU_Disable(struct ethosu_driver *drv);






void ETHOSU_PMU_Set_EVTYPER(struct ethosu_driver *drv, uint32_t num, enum ethosu_pmu_event_type type);





uint32_t ETHOSU_PMU_Get_NumEventCounters(void);






enum ethosu_pmu_event_type ETHOSU_PMU_Get_EVTYPER(struct ethosu_driver *drv, uint32_t num);




void ETHOSU_PMU_CYCCNT_Reset(struct ethosu_driver *drv);




void ETHOSU_PMU_EVCNTR_ALL_Reset(struct ethosu_driver *drv);
# 186 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
void ETHOSU_PMU_CNTR_Enable(struct ethosu_driver *drv, uint32_t mask);
# 195 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
void ETHOSU_PMU_CNTR_Disable(struct ethosu_driver *drv, uint32_t mask);
# 207 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
uint32_t ETHOSU_PMU_CNTR_Status(struct ethosu_driver *drv);
# 218 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
uint64_t ETHOSU_PMU_Get_CCNTR(struct ethosu_driver *drv);
# 227 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
void ETHOSU_PMU_Set_CCNTR(struct ethosu_driver *drv, uint64_t val);






uint32_t ETHOSU_PMU_Get_EVCNTR(struct ethosu_driver *drv, uint32_t num);







void ETHOSU_PMU_Set_EVCNTR(struct ethosu_driver *drv, uint32_t num, uint32_t val);







uint32_t ETHOSU_PMU_Get_CNTR_OVS(struct ethosu_driver *drv);
# 259 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
void ETHOSU_PMU_Set_CNTR_OVS(struct ethosu_driver *drv, uint32_t mask);
# 268 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
void ETHOSU_PMU_Set_CNTR_IRQ_Enable(struct ethosu_driver *drv, uint32_t mask);
# 277 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
void ETHOSU_PMU_Set_CNTR_IRQ_Disable(struct ethosu_driver *drv, uint32_t mask);
# 287 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
uint32_t ETHOSU_PMU_Get_IRQ_Enable(struct ethosu_driver *drv);
# 296 "../../../../Library/StdDriver/inc/npu\\pmu_ethosu.h"
void ETHOSU_PMU_CNTR_Increment(struct ethosu_driver *drv, uint32_t mask);






void ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(struct ethosu_driver *drv, enum ethosu_pmu_event_type start_event);






void ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(struct ethosu_driver *drv, enum ethosu_pmu_event_type stop_event);




uint32_t ETHOSU_PMU_Get_QREAD(struct ethosu_driver *drv);




uint32_t ETHOSU_PMU_Get_STATUS(struct ethosu_driver *drv);
# 21 "../NPU/include\\ethosu_profiler.h" 2







typedef struct npu_event_counter_
{
    enum ethosu_pmu_event_type event_type;
    uint32_t event_mask;
    uint32_t counter_value;
    char *unit;
    char *name;
} npu_evt_counter;

typedef struct npu_derived_counter_
{
    uint32_t counter_value;
    char *unit;
    char *name;
} npu_derived_counter;

typedef struct ethosu_pmu_counters_
{
    uint64_t npu_total_ccnt;
    npu_evt_counter npu_evt_counters[4];
    npu_derived_counter npu_derived_counters[1];
    uint32_t num_total_counters;
} ethosu_pmu_counters;




void ethosu_pmu_init(void);




void ethosu_pmu_reset_counters(void);





ethosu_pmu_counters ethosu_get_pmu_counters(void);
# 19 "../NPU/ethosu_profiler.c" 2
# 1 "../../../../ThirdParty/ml-embedded-evaluation-kit/source/log/include\\log_macros.h" 1
# 24 "../../../../ThirdParty/ml-embedded-evaluation-kit/source/log/include\\log_macros.h"
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 1 3
# 68 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
    typedef __builtin_va_list __va_list;
# 87 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
typedef struct __fpos_t_struct {
    unsigned long long int __pos;





    struct {
        unsigned int __state1, __state2;
    } __mbstate;
} fpos_t;
# 108 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
typedef struct __FILE FILE;
# 119 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
struct __FILE {
    union {
        long __FILE_alignment;



        char __FILE_size[84];

    } __FILE_opaque;
};
# 138 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern FILE __stdin, __stdout, __stderr;
extern FILE *__aeabi_stdin, *__aeabi_stdout, *__aeabi_stderr;
# 224 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int remove(const char * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int rename(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 243 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) FILE *tmpfile(void);






extern __attribute__((__nothrow__)) char *tmpnam(char * );
# 265 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fclose(FILE * ) __attribute__((__nonnull__(1)));
# 275 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fflush(FILE * );
# 285 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) FILE *fopen(const char * __restrict ,
                           const char * __restrict ) __attribute__((__nonnull__(1,2)));
# 329 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) FILE *freopen(const char * __restrict ,
                    const char * __restrict ,
                    FILE * __restrict ) __attribute__((__nonnull__(2,3)));
# 342 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) void setbuf(FILE * __restrict ,
                    char * __restrict ) __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) int setvbuf(FILE * __restrict ,
                   char * __restrict ,
                   int , size_t ) __attribute__((__nonnull__(1)));
# 370 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int fprintf(FILE * __restrict ,
                    const char * __restrict , ...) __attribute__((__nonnull__(1,2)));
# 393 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
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
# 460 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int _snprintf(char * __restrict , size_t ,
                      const char * __restrict , ...) __attribute__((__nonnull__(3)));





#pragma __scanf_args
extern __attribute__((__nothrow__)) int fscanf(FILE * __restrict ,
                    const char * __restrict , ...) __attribute__((__nonnull__(1,2)));
# 503 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
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
# 541 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
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
# 584 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int vsprintf(char * __restrict ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));
# 594 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int __ARM_vsnprintf(char * __restrict , size_t ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(3)));

extern __attribute__((__nothrow__)) int vsnprintf(char * __restrict , size_t ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(3)));
# 609 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int _vsprintf(char * __restrict ,
                      const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));





extern __attribute__((__nothrow__)) int _vfprintf(FILE * __restrict ,
                     const char * __restrict , __va_list ) __attribute__((__nonnull__(1,2)));





extern __attribute__((__nothrow__)) int _vsnprintf(char * __restrict , size_t ,
                      const char * __restrict , __va_list ) __attribute__((__nonnull__(3)));
# 635 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
#pragma __printf_args
extern __attribute__((__nothrow__)) int __ARM_asprintf(char ** , const char * __restrict , ...) __attribute__((__nonnull__(2)));
extern __attribute__((__nothrow__)) int __ARM_vasprintf(char ** , const char * __restrict , __va_list ) __attribute__((__nonnull__(2)));
# 649 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fgetc(FILE * ) __attribute__((__nonnull__(1)));
# 659 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) char *fgets(char * __restrict , int ,
                    FILE * __restrict ) __attribute__((__nonnull__(1,3)));
# 673 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fputc(int , FILE * ) __attribute__((__nonnull__(2)));
# 683 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fputs(const char * __restrict , FILE * __restrict ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) int getc(FILE * ) __attribute__((__nonnull__(1)));
# 704 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
    extern __attribute__((__nothrow__)) int (getchar)(void);
# 713 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) char *gets(char * ) __attribute__((__nonnull__(1)));
# 725 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int putc(int , FILE * ) __attribute__((__nonnull__(2)));
# 737 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
    extern __attribute__((__nothrow__)) int (putchar)(int );






extern __attribute__((__nothrow__)) int puts(const char * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int ungetc(int , FILE * ) __attribute__((__nonnull__(2)));
# 778 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) size_t fread(void * __restrict ,
                    size_t , size_t , FILE * __restrict ) __attribute__((__nonnull__(1,4)));
# 794 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) size_t __fread_bytes_avail(void * __restrict ,
                    size_t , FILE * __restrict ) __attribute__((__nonnull__(1,3)));
# 810 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) size_t fwrite(const void * __restrict ,
                    size_t , size_t , FILE * __restrict ) __attribute__((__nonnull__(1,4)));
# 822 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fgetpos(FILE * __restrict , fpos_t * __restrict ) __attribute__((__nonnull__(1,2)));
# 833 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fseek(FILE * , long int , int ) __attribute__((__nonnull__(1)));
# 850 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int fsetpos(FILE * __restrict , const fpos_t * __restrict ) __attribute__((__nonnull__(1,2)));
# 863 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) long int ftell(FILE * ) __attribute__((__nonnull__(1)));
# 877 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) void rewind(FILE * ) __attribute__((__nonnull__(1)));
# 886 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) void clearerr(FILE * ) __attribute__((__nonnull__(1)));







extern __attribute__((__nothrow__)) int feof(FILE * ) __attribute__((__nonnull__(1)));




extern __attribute__((__nothrow__)) int ferror(FILE * ) __attribute__((__nonnull__(1)));




extern __attribute__((__nothrow__)) void perror(const char * );
# 917 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\stdio.h" 3
extern __attribute__((__nothrow__)) int _fisatty(FILE * ) __attribute__((__nonnull__(1)));



extern __attribute__((__nothrow__)) void __use_no_semihosting_swi(void);
extern __attribute__((__nothrow__)) void __use_no_semihosting(void);
# 25 "../../../../ThirdParty/ml-embedded-evaluation-kit/source/log/include\\log_macros.h" 2
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\inttypes.h" 1 3
# 222 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\inttypes.h" 3
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
# 26 "../../../../ThirdParty/ml-embedded-evaluation-kit/source/log/include\\log_macros.h" 2
# 20 "../NPU/ethosu_profiler.c" 2

# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 1 3
# 58 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void *memcpy(void * __restrict ,
                    const void * __restrict , size_t ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) void *memmove(void * ,
                    const void * , size_t ) __attribute__((__nonnull__(1,2)));
# 77 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strcpy(char * __restrict , const char * __restrict ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) char *strncpy(char * __restrict , const char * __restrict , size_t ) __attribute__((__nonnull__(1,2)));
# 93 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strcat(char * __restrict , const char * __restrict ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) char *strncat(char * __restrict , const char * __restrict , size_t ) __attribute__((__nonnull__(1,2)));
# 117 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) int memcmp(const void * , const void * , size_t ) __attribute__((__nonnull__(1,2)));







extern __attribute__((__nothrow__)) int strcmp(const char * , const char * ) __attribute__((__nonnull__(1,2)));






extern __attribute__((__nothrow__)) int strncmp(const char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 141 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) int strcasecmp(const char * , const char * ) __attribute__((__nonnull__(1,2)));







extern __attribute__((__nothrow__)) int strncasecmp(const char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 158 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) int strcoll(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 169 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strxfrm(char * __restrict , const char * __restrict , size_t ) __attribute__((__nonnull__(2)));
# 193 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void *memchr(const void * , int , size_t ) __attribute__((__nonnull__(1)));
# 209 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strchr(const char * , int ) __attribute__((__nonnull__(1)));
# 218 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strcspn(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 232 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strpbrk(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 247 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strrchr(const char * , int ) __attribute__((__nonnull__(1)));
# 257 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strspn(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 270 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strstr(const char * , const char * ) __attribute__((__nonnull__(1,2)));
# 280 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) char *strtok(char * __restrict , const char * __restrict ) __attribute__((__nonnull__(2)));
extern __attribute__((__nothrow__)) char *_strtok_r(char * , const char * , char ** ) __attribute__((__nonnull__(2,3)));
# 321 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) void *memset(void * , int , size_t ) __attribute__((__nonnull__(1)));





extern __attribute__((__nothrow__)) char *strerror(int );







extern __attribute__((__nothrow__)) size_t strlen(const char * ) __attribute__((__nonnull__(1)));






extern __attribute__((__nothrow__)) size_t strlcpy(char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 362 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
extern __attribute__((__nothrow__)) size_t strlcat(char * , const char * , size_t ) __attribute__((__nonnull__(1,2)));
# 388 "C:\\Keil_v5\\ARM\\ARMCLANG\\Bin\\..\\include\\string.h" 3
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
# 22 "../NPU/ethosu_profiler.c" 2

extern struct ethosu_driver ethosu_drv;
static ethosu_pmu_counters npu_counters;





static ethosu_pmu_counters *get_counter_instance(void)
{
    return &npu_counters;
}






static _Bool counter_overflow(uint32_t pmu_counter_mask)
{


    const uint32_t overflow_status = ETHOSU_PMU_Get_CNTR_OVS(&ethosu_drv);
    return pmu_counter_mask & overflow_status ? 1 : 0;
}

void ethosu_pmu_init(void)
{
    uint32_t i = 0;
    uint32_t evt_mask = (1UL << 31);
    ethosu_pmu_counters *counters = get_counter_instance();
    memset(counters, 0, sizeof(*counters));


    counters->num_total_counters = ( 1 + 4 + 1 );


    counters->npu_evt_counters[0].event_type = ETHOSU_PMU_NPU_IDLE;
    counters->npu_evt_counters[0].event_mask = (1UL << 0);
    counters->npu_evt_counters[0].name = "NPU IDLE";
    counters->npu_evt_counters[0].unit = "cycles";

    counters->npu_evt_counters[1].event_type = ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED;
    counters->npu_evt_counters[1].event_mask = (1UL << 1);
    counters->npu_evt_counters[1].name = "NPU AXI0_RD_DATA_BEAT_RECEIVED";
    counters->npu_evt_counters[1].unit = "beats";

    counters->npu_evt_counters[2].event_type = ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN;
    counters->npu_evt_counters[2].event_mask = (1UL << 2);
    counters->npu_evt_counters[2].name = "NPU AXI0_WR_DATA_BEAT_WRITTEN";
    counters->npu_evt_counters[2].unit = "beats";

    counters->npu_evt_counters[3].event_type = ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED;
    counters->npu_evt_counters[3].event_mask = (1UL << 3);
    counters->npu_evt_counters[3].name = "NPU AXI1_RD_DATA_BEAT_RECEIVED";
    counters->npu_evt_counters[3].unit = "beats";





    counters->npu_derived_counters[0].name = "NPU ACTIVE";
    counters->npu_derived_counters[0].unit = "cycles";



    ETHOSU_PMU_Enable(&ethosu_drv);

    for (i = 0; i < 4; ++i)
    {
        ETHOSU_PMU_Set_EVTYPER(&ethosu_drv, i, counters->npu_evt_counters[i].event_type);
        evt_mask |= counters->npu_evt_counters[i].event_mask;
    }


    ETHOSU_PMU_Set_CNTR_OVS(&ethosu_drv, evt_mask);


    ETHOSU_PMU_CNTR_Disable(&ethosu_drv, evt_mask);
    ethosu_pmu_reset_counters();
    ETHOSU_PMU_CNTR_Enable(&ethosu_drv, evt_mask);
}




void ethosu_pmu_reset_counters(void)
{

    ETHOSU_PMU_CYCCNT_Reset(&ethosu_drv);
    ETHOSU_PMU_EVCNTR_ALL_Reset(&ethosu_drv);
}





ethosu_pmu_counters ethosu_get_pmu_counters(void)
{
    ethosu_pmu_counters *counters = get_counter_instance();
    uint32_t i = 0;


    for (i = 0; i < 4; ++i)
    {
        if (counter_overflow(counters->npu_evt_counters[i].event_mask))
        {
            printf("WARN - "); printf("Counter overflow detected for %s.\n", counters->npu_evt_counters[i].name);
        }

        counters->npu_evt_counters[i].counter_value =
            ETHOSU_PMU_Get_EVCNTR(&ethosu_drv, i);
    }


    counters->npu_total_ccnt = ETHOSU_PMU_Get_CCNTR(&ethosu_drv);




    if (counters->npu_evt_counters[0].event_type == ETHOSU_PMU_NPU_IDLE)
    {
        counters->npu_derived_counters[0].counter_value =
            counters->npu_total_ccnt - counters->npu_evt_counters[0].counter_value;
    }



    return *counters;
}
