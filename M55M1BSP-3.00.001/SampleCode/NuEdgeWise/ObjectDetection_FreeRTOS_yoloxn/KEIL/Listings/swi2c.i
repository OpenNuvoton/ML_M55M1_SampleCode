# 1 "../Device/ImageSensor/Sensor/SWI2C.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 407 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "../Device/ImageSensor/Sensor/SWI2C.c" 2








# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 1 3
# 71 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
    typedef unsigned int size_t;
# 91 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdlib.h" 3
    typedef unsigned short wchar_t;




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
# 10 "../Device/ImageSensor/Sensor/SWI2C.c" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\NuMicro.h" 1
# 14 "../../../../Library/Device/Nuvoton/M55M1/Include\\NuMicro.h"
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 1
# 79 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h"
typedef enum IRQn
{

    NonMaskableInt_IRQn = -14,
    HardFault_IRQn = -13,
    MemoryManagement_IRQn = -12,
    BusFault_IRQn = -11,
    UsageFault_IRQn = -10,
    SecureFault_IRQn = -9,
    SVCall_IRQn = -5,
    DebugMonitor_IRQn = -4,
    PendSV_IRQn = -2,
    SysTick_IRQn = -1,


    BODOUT_IRQn = 0,
    IRC_IRQn = 1,
    PMC_IRQn = 2,
    SRAMPERR_IRQn = 3,
    CKFAIL_IRQn = 4,
    ISP_IRQn = 5,
    RTC_IRQn = 6,
    RTCTAMPER_IRQn = 7,
    WDT0_IRQn = 8,
    WWDT0_IRQn = 9,


    WDT1_IRQn = 11,
    WWDT1_IRQn = 12,
    NPU_IRQn = 13,
    EINT0_IRQn = 14,
    EINT1_IRQn = 15,
    EINT2_IRQn = 16,
    EINT3_IRQn = 17,
    EINT4_IRQn = 18,
    EINT5_IRQn = 19,

    GPA_IRQn = 20,
    GPB_IRQn = 21,
    GPC_IRQn = 22,
    GPD_IRQn = 23,
    GPE_IRQn = 24,
    GPF_IRQn = 25,
    GPG_IRQn = 26,
    GPH_IRQn = 27,
    GPI_IRQn = 28,
    GPJ_IRQn = 29,

    BRAKE0_IRQn = 30,
    EPWM0P0_IRQn = 31,
    EPWM0P1_IRQn = 32,
    EPWM0P2_IRQn = 33,
    BRAKE1_IRQn = 34,
    EPWM1P0_IRQn = 35,
    EPWM1P1_IRQn = 36,
    EPWM1P2_IRQn = 37,
    BPWM0_IRQn = 38,
    BPWM1_IRQn = 39,


    PDMA0_IRQn = 41,
    PDMA1_IRQn = 42,
    LPPDMA_IRQn = 43,
    SCU_IRQn = 44,

    KS_IRQn = 46,
    TIMER0_IRQn = 47,
    TIMER1_IRQn = 48,
    TIMER2_IRQn = 49,

    TIMER3_IRQn = 50,
    LPTMR0_IRQn = 51,
    LPTMR1_IRQn = 52,

    TTMR0_IRQn = 54,
    TTMR1_IRQn = 55,
    USBH0_IRQn = 56,
    USBH1_IRQn = 57,
    USBD_IRQn = 58,
    USBOTG_IRQn = 59,

    HSUSBH_IRQn = 60,
    HSUSBD_IRQn = 61,
    HSOTG_IRQn = 62,
    EMAC0_IRQn = 63,
    QSPI0_IRQn = 64,
    QSPI1_IRQn = 65,
    SPI0_IRQn = 66,
    SPI1_IRQn = 67,
    SPI2_IRQn = 68,
    SPI3_IRQn = 69,


    LPSPI0_IRQn = 71,

    SPIM0_IRQn = 73,

    SPIM1_IRQn = 74,

    UART0_IRQn = 75,
    UART1_IRQn = 76,
    UART2_IRQn = 77,
    UART3_IRQn = 78,
    UART4_IRQn = 79,

    UART5_IRQn = 80,
    UART6_IRQn = 81,
    UART7_IRQn = 82,
    UART8_IRQn = 83,
    UART9_IRQn = 84,



    EINT6_IRQn = 88,
    EINT7_IRQn = 89,

    LPUART0_IRQn = 90,

    I2C0_IRQn = 92,
    I2C1_IRQn = 93,
    I2C2_IRQn = 94,
    I2C3_IRQn = 95,
    LPI2C0_IRQn = 96,
    USCI0_IRQn = 97,

    SC0_IRQn = 99,

    SC1_IRQn = 100,
    SC2_IRQn = 101,
    PSIO_IRQn = 102,
    DMIC0_IRQn = 103,
    DMIC0VAD_IRQn = 104,
    I2S0_IRQn = 105,
    I2S1_IRQn = 106,
    TRNG_IRQn = 107,
    I3C0_IRQn = 108,


    OTFC0_IRQn = 110,

    OTFC1_IRQn = 111,


    KPI_IRQn = 112,
    SDH0_IRQn = 113,
    SDH1_IRQn = 114,
    CCAP_IRQn = 115,
    CRYPTO_IRQn = 116,
    CANFD00_IRQn = 117,
    CANFD01_IRQn = 118,
    CANFD10_IRQn = 119,

    CANFD11_IRQn = 120,
    ACMP01_IRQn = 121,
    ACMP23_IRQn = 122,


    CRC_IRQn = 125,
    EADC00_IRQn = 126,
    EADC01_IRQn = 127,
    EADC02_IRQn = 128,
    EADC03_IRQn = 129,





    LPADC0_IRQn = 134,

    DAC01_IRQn = 135,

    EQEI0_IRQn = 137,
    EQEI1_IRQn = 138,
    EQEI2_IRQn = 139,

    EQEI3_IRQn = 140,
    ECAP0_IRQn = 141,
    ECAP1_IRQn = 142,
    ECAP2_IRQn = 143,
    ECAP3_IRQn = 144,




    AWF_IRQn = 149,

    UTCPD_IRQn = 150,
# 276 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h"
    GDMACH0_IRQn = 160,
    GDMACH1_IRQn = 161,
    TOTAL_IRQn_CNT = GDMACH1_IRQn + 16,
} IRQn_Type;
# 327 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h"
# 1 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 1
# 29 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3







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
# 37 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 2 3
# 65 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
# 1 "../../../../Library/CMSIS/Core/Include\\cmsis_version.h" 1 3
# 29 "../../../../Library/CMSIS/Core/Include\\cmsis_version.h" 3
# 66 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 2 3
# 184 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
# 1 "../../../../Library/CMSIS/Core/Include\\cmsis_compiler.h" 1 3
# 47 "../../../../Library/CMSIS/Core/Include\\cmsis_compiler.h" 3
# 1 "../../../../Library/CMSIS/Core/Include\\cmsis_armclang.h" 1 3
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
# 48 "../../../../Library/CMSIS/Core/Include\\cmsis_compiler.h" 2 3
# 185 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 2 3
# 330 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef union
{
  struct
  {
    uint32_t _reserved0:16;
    uint32_t GE:4;
    uint32_t _reserved1:7;
    uint32_t Q:1;
    uint32_t V:1;
    uint32_t C:1;
    uint32_t Z:1;
    uint32_t N:1;
  } b;
  uint32_t w;
} APSR_Type;
# 369 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef union
{
  struct
  {
    uint32_t ISR:9;
    uint32_t _reserved0:23;
  } b;
  uint32_t w;
} IPSR_Type;
# 387 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef union
{
  struct
  {
    uint32_t ISR:9;
    uint32_t _reserved0:7;
    uint32_t GE:4;
    uint32_t _reserved1:4;
    uint32_t T:1;
    uint32_t IT:2;
    uint32_t Q:1;
    uint32_t V:1;
    uint32_t C:1;
    uint32_t Z:1;
    uint32_t N:1;
  } b;
  uint32_t w;
} xPSR_Type;
# 438 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef union
{
  struct
  {
    uint32_t nPRIV:1;
    uint32_t SPSEL:1;
    uint32_t FPCA:1;
    uint32_t SFPA:1;
    uint32_t _reserved1:28;
  } b;
  uint32_t w;
} CONTROL_Type;
# 477 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t ISER[16U];
        uint32_t RESERVED0[16U];
  volatile uint32_t ICER[16U];
        uint32_t RSERVED1[16U];
  volatile uint32_t ISPR[16U];
        uint32_t RESERVED2[16U];
  volatile uint32_t ICPR[16U];
        uint32_t RESERVED3[16U];
  volatile uint32_t IABR[16U];
        uint32_t RESERVED4[16U];
  volatile uint32_t ITNS[16U];
        uint32_t RESERVED5[16U];
  volatile uint8_t IPR[496U];
        uint32_t RESERVED6[580U];
  volatile uint32_t STIR;
} NVIC_Type;
# 513 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile const uint32_t CPUID;
  volatile uint32_t ICSR;
  volatile uint32_t VTOR;
  volatile uint32_t AIRCR;
  volatile uint32_t SCR;
  volatile uint32_t CCR;
  volatile uint8_t SHPR[12U];
  volatile uint32_t SHCSR;
  volatile uint32_t CFSR;
  volatile uint32_t HFSR;
  volatile uint32_t DFSR;
  volatile uint32_t MMFAR;
  volatile uint32_t BFAR;
  volatile uint32_t AFSR;
  volatile const uint32_t ID_PFR[2U];
  volatile const uint32_t ID_DFR;
  volatile const uint32_t ID_AFR;
  volatile const uint32_t ID_MMFR[4U];
  volatile const uint32_t ID_ISAR[6U];
  volatile const uint32_t CLIDR;
  volatile const uint32_t CTR;
  volatile const uint32_t CCSIDR;
  volatile uint32_t CSSELR;
  volatile uint32_t CPACR;
  volatile uint32_t NSACR;
        uint32_t RESERVED7[21U];
  volatile uint32_t SFSR;
  volatile uint32_t SFAR;
        uint32_t RESERVED3[69U];
  volatile uint32_t STIR;
  volatile uint32_t RFSR;
        uint32_t RESERVED4[14U];
  volatile const uint32_t MVFR0;
  volatile const uint32_t MVFR1;
  volatile const uint32_t MVFR2;
        uint32_t RESERVED5[1U];
  volatile uint32_t ICIALLU;
        uint32_t RESERVED6[1U];
  volatile uint32_t ICIMVAU;
  volatile uint32_t DCIMVAC;
  volatile uint32_t DCISW;
  volatile uint32_t DCCMVAU;
  volatile uint32_t DCCMVAC;
  volatile uint32_t DCCSW;
  volatile uint32_t DCCIMVAC;
  volatile uint32_t DCCISW;
  volatile uint32_t BPIALL;
} SCB_Type;
# 1003 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
        uint32_t RESERVED0[1U];
  volatile const uint32_t ICTR;
  volatile uint32_t ACTLR;
  volatile uint32_t CPPWR;
} ICB_Type;
# 1071 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t CTRL;
  volatile uint32_t LOAD;
  volatile uint32_t VAL;
  volatile const uint32_t CALIB;
} SysTick_Type;
# 1123 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile union
  {
    volatile uint8_t u8;
    volatile uint16_t u16;
    volatile uint32_t u32;
  } PORT [32U];
        uint32_t RESERVED0[864U];
  volatile uint32_t TER;
        uint32_t RESERVED1[15U];
  volatile uint32_t TPR;
        uint32_t RESERVED2[15U];
  volatile uint32_t TCR;
        uint32_t RESERVED3[32U];
        uint32_t RESERVED4[43U];
  volatile uint32_t LAR;
  volatile const uint32_t LSR;
        uint32_t RESERVED5[1U];
  volatile const uint32_t DEVARCH;
        uint32_t RESERVED6[3U];
  volatile const uint32_t DEVTYPE;
  volatile const uint32_t PID4;
  volatile const uint32_t PID5;
  volatile const uint32_t PID6;
  volatile const uint32_t PID7;
  volatile const uint32_t PID0;
  volatile const uint32_t PID1;
  volatile const uint32_t PID2;
  volatile const uint32_t PID3;
  volatile const uint32_t CID0;
  volatile const uint32_t CID1;
  volatile const uint32_t CID2;
  volatile const uint32_t CID3;
} ITM_Type;
# 1224 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t CTRL;
  volatile uint32_t CYCCNT;
  volatile uint32_t CPICNT;
  volatile uint32_t EXCCNT;
  volatile uint32_t SLEEPCNT;
  volatile uint32_t LSUCNT;
  volatile uint32_t FOLDCNT;
  volatile const uint32_t PCSR;
  volatile uint32_t COMP0;
        uint32_t RESERVED1[1U];
  volatile uint32_t FUNCTION0;
        uint32_t RESERVED2[1U];
  volatile uint32_t COMP1;
        uint32_t RESERVED3[1U];
  volatile uint32_t FUNCTION1;
        uint32_t RESERVED4[1U];
  volatile uint32_t COMP2;
        uint32_t RESERVED5[1U];
  volatile uint32_t FUNCTION2;
        uint32_t RESERVED6[1U];
  volatile uint32_t COMP3;
        uint32_t RESERVED7[1U];
  volatile uint32_t FUNCTION3;
        uint32_t RESERVED8[1U];
  volatile uint32_t COMP4;
        uint32_t RESERVED9[1U];
  volatile uint32_t FUNCTION4;
        uint32_t RESERVED10[1U];
  volatile uint32_t COMP5;
        uint32_t RESERVED11[1U];
  volatile uint32_t FUNCTION5;
        uint32_t RESERVED12[1U];
  volatile uint32_t COMP6;
        uint32_t RESERVED13[1U];
  volatile uint32_t FUNCTION6;
        uint32_t RESERVED14[1U];
  volatile uint32_t COMP7;
        uint32_t RESERVED15[1U];
  volatile uint32_t FUNCTION7;
        uint32_t RESERVED16[1U];
  volatile uint32_t COMP8;
        uint32_t RESERVED17[1U];
  volatile uint32_t FUNCTION8;
        uint32_t RESERVED18[1U];
  volatile uint32_t COMP9;
        uint32_t RESERVED19[1U];
  volatile uint32_t FUNCTION9;
        uint32_t RESERVED20[1U];
  volatile uint32_t COMP10;
        uint32_t RESERVED21[1U];
  volatile uint32_t FUNCTION10;
        uint32_t RESERVED22[1U];
  volatile uint32_t COMP11;
        uint32_t RESERVED23[1U];
  volatile uint32_t FUNCTION11;
        uint32_t RESERVED24[1U];
  volatile uint32_t COMP12;
        uint32_t RESERVED25[1U];
  volatile uint32_t FUNCTION12;
        uint32_t RESERVED26[1U];
  volatile uint32_t COMP13;
        uint32_t RESERVED27[1U];
  volatile uint32_t FUNCTION13;
        uint32_t RESERVED28[1U];
  volatile uint32_t COMP14;
        uint32_t RESERVED29[1U];
  volatile uint32_t FUNCTION14;
        uint32_t RESERVED30[1U];
  volatile uint32_t COMP15;
        uint32_t RESERVED31[1U];
  volatile uint32_t FUNCTION15;
        uint32_t RESERVED32[934U];
  volatile const uint32_t LSR;
        uint32_t RESERVED33[1U];
  volatile const uint32_t DEVARCH;
} DWT_Type;
# 1410 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t MSCR;
  volatile uint32_t PFCR;
        uint32_t RESERVED1[2U];
  volatile uint32_t ITCMCR;
  volatile uint32_t DTCMCR;
  volatile uint32_t PAHBCR;
        uint32_t RESERVED2[313U];
  volatile uint32_t ITGU_CTRL;
  volatile uint32_t ITGU_CFG;
        uint32_t RESERVED3[2U];
  volatile uint32_t ITGU_LUT[16U];
        uint32_t RESERVED4[44U];
  volatile uint32_t DTGU_CTRL;
  volatile uint32_t DTGU_CFG;
        uint32_t RESERVED5[2U];
  volatile uint32_t DTGU_LUT[16U];
} MemSysCtl_Type;
# 1537 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t CPDLPSTATE;
  volatile uint32_t DPDLPSTATE;
} PwrModCtl_Type;
# 1570 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t EVENTSPR;
        uint32_t RESERVED0[31U];
  volatile const uint32_t EVENTMASKA;
  volatile const uint32_t EVENTMASK[15];
} EWIC_Type;
# 1615 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t IEBR0;
  volatile uint32_t IEBR1;
        uint32_t RESERVED0[2U];
  volatile uint32_t DEBR0;
  volatile uint32_t DEBR1;
        uint32_t RESERVED1[2U];
  volatile uint32_t TEBR0;
        uint32_t RESERVED2[1U];
  volatile uint32_t TEBR1;
} ErrBnk_Type;
# 1755 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t CFGINFOSEL;
  volatile const uint32_t CFGINFORD;
} PrcCfgInf_Type;
# 1778 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile const uint32_t STLNVICPENDOR;
  volatile const uint32_t STLNVICACTVOR;
        uint32_t RESERVED0[2U];
  volatile uint32_t STLIDMPUSR;
  volatile const uint32_t STLIMPUOR;
  volatile const uint32_t STLD0MPUOR;
  volatile const uint32_t STLD1MPUOR;

} STL_Type;
# 1860 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile const uint32_t SSPSR;
  volatile uint32_t CSPSR;
        uint32_t RESERVED0[2U];
  volatile uint32_t ACPR;
        uint32_t RESERVED1[55U];
  volatile uint32_t SPPR;
        uint32_t RESERVED2[131U];
  volatile const uint32_t FFSR;
  volatile uint32_t FFCR;
  volatile uint32_t PSCR;
        uint32_t RESERVED3[809U];
  volatile uint32_t LAR;
  volatile const uint32_t LSR;
        uint32_t RESERVED4[4U];
  volatile const uint32_t TYPE;
  volatile const uint32_t DEVTYPE;
} TPI_Type;
# 1958 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t EVCNTR[8U];

        uint32_t RESERVED0[31U-8U];

  volatile uint32_t CCNTR;
        uint32_t RESERVED1[224];
  volatile uint32_t EVTYPER[8U];

        uint32_t RESERVED2[31U-8U];

  volatile uint32_t CCFILTR;
        uint32_t RESERVED3[480];
  volatile uint32_t CNTENSET;
        uint32_t RESERVED4[7];
  volatile uint32_t CNTENCLR;
        uint32_t RESERVED5[7];
  volatile uint32_t INTENSET;
        uint32_t RESERVED6[7];
  volatile uint32_t INTENCLR;
        uint32_t RESERVED7[7];
  volatile uint32_t OVSCLR;
        uint32_t RESERVED8[7];
  volatile uint32_t SWINC;
        uint32_t RESERVED9[7];
  volatile uint32_t OVSSET;
        uint32_t RESERVED10[79];
  volatile uint32_t TYPE;
  volatile uint32_t CTRL;
        uint32_t RESERVED11[108];
  volatile uint32_t AUTHSTATUS;
  volatile uint32_t DEVARCH;
        uint32_t RESERVED12[3];
  volatile uint32_t DEVTYPE;
  volatile uint32_t PIDR4;
        uint32_t RESERVED13[3];
  volatile uint32_t PIDR0;
  volatile uint32_t PIDR1;
  volatile uint32_t PIDR2;
  volatile uint32_t PIDR3;
  volatile uint32_t CIDR0;
  volatile uint32_t CIDR1;
  volatile uint32_t CIDR2;
  volatile uint32_t CIDR3;
} PMU_Type;
# 2776 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile const uint32_t TYPE;
  volatile uint32_t CTRL;
  volatile uint32_t RNR;
  volatile uint32_t RBAR;
  volatile uint32_t RLAR;
  volatile uint32_t RBAR_A1;
  volatile uint32_t RLAR_A1;
  volatile uint32_t RBAR_A2;
  volatile uint32_t RLAR_A2;
  volatile uint32_t RBAR_A3;
  volatile uint32_t RLAR_A3;
        uint32_t RESERVED0[1];
  union {
  volatile uint32_t MAIR[2];
  struct {
  volatile uint32_t MAIR0;
  volatile uint32_t MAIR1;
  };
  };
} MPU_Type;
# 2978 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
        uint32_t RESERVED0[1U];
  volatile uint32_t FPCCR;
  volatile uint32_t FPCAR;
  volatile uint32_t FPDSCR;
  volatile const uint32_t MVFR0;
  volatile const uint32_t MVFR1;
  volatile const uint32_t MVFR2;
} FPU_Type;
# 3119 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t DHCSR;
  volatile uint32_t DCRSR;
  volatile uint32_t DCRDR;
  volatile uint32_t DEMCR;
  volatile uint32_t DSCEMCR;
  volatile uint32_t DAUTHCTRL;
  volatile uint32_t DSCSR;
} CoreDebug_Type;
# 3290 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t DHCSR;
  volatile uint32_t DCRSR;
  volatile uint32_t DCRDR;
  volatile uint32_t DEMCR;
  volatile uint32_t DSCEMCR;
  volatile uint32_t DAUTHCTRL;
  volatile uint32_t DSCSR;
} DCB_Type;
# 3481 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef struct
{
  volatile uint32_t DLAR;
  volatile const uint32_t DLSR;
  volatile const uint32_t DAUTHSTATUS;
  volatile const uint32_t DDEVARCH;
  volatile const uint32_t DDEVTYPE;
} DIB_Type;
# 3680 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
typedef ICB_Type SCnSCB_Type;
# 3821 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline void __NVIC_SetPriorityGrouping(uint32_t PriorityGroup)
{
  uint32_t reg_value;
  uint32_t PriorityGroupTmp = (PriorityGroup & (uint32_t)0x07UL);

  reg_value = ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->AIRCR;
  reg_value &= ~((uint32_t)((0xFFFFUL << 16U) | (7UL << 8U)));
  reg_value = (reg_value |
                ((uint32_t)0x5FAUL << 16U) |
                (PriorityGroupTmp << 8U) );
  ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->AIRCR = reg_value;
}







static __inline uint32_t __NVIC_GetPriorityGrouping(void)
{
  return ((uint32_t)((((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->AIRCR & (7UL << 8U)) >> 8U));
}
# 3852 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline void __NVIC_EnableIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    __asm volatile("":::"memory");
    ((NVIC_Type *) ((0xE000E000UL) + 0x0100UL) )->ISER[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
    __asm volatile("":::"memory");
  }
}
# 3871 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t __NVIC_GetEnableIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((((NVIC_Type *) ((0xE000E000UL) + 0x0100UL) )->ISER[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}
# 3890 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline void __NVIC_DisableIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    ((NVIC_Type *) ((0xE000E000UL) + 0x0100UL) )->ICER[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);
  }
}
# 3909 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t __NVIC_GetPendingIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((((NVIC_Type *) ((0xE000E000UL) + 0x0100UL) )->ISPR[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}
# 3928 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline void __NVIC_SetPendingIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    ((NVIC_Type *) ((0xE000E000UL) + 0x0100UL) )->ISPR[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}
# 3943 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline void __NVIC_ClearPendingIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    ((NVIC_Type *) ((0xE000E000UL) + 0x0100UL) )->ICPR[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}
# 3960 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t __NVIC_GetActive(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((((NVIC_Type *) ((0xE000E000UL) + 0x0100UL) )->IABR[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}
# 4049 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline void __NVIC_SetPriority(IRQn_Type IRQn, uint32_t priority)
{
  if ((int32_t)(IRQn) >= 0)
  {
    ((NVIC_Type *) ((0xE000E000UL) + 0x0100UL) )->IPR[((uint32_t)IRQn)] = (uint8_t)((priority << (8U - 3U)) & (uint32_t)0xFFUL);
  }
  else
  {
    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->SHPR[(((uint32_t)IRQn) & 0xFUL)-4UL] = (uint8_t)((priority << (8U - 3U)) & (uint32_t)0xFFUL);
  }
}
# 4071 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t __NVIC_GetPriority(IRQn_Type IRQn)
{

  if ((int32_t)(IRQn) >= 0)
  {
    return(((uint32_t)((NVIC_Type *) ((0xE000E000UL) + 0x0100UL) )->IPR[((uint32_t)IRQn)] >> (8U - 3U)));
  }
  else
  {
    return(((uint32_t)((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->SHPR[(((uint32_t)IRQn) & 0xFUL)-4UL] >> (8U - 3U)));
  }
}
# 4096 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t NVIC_EncodePriority (uint32_t PriorityGroup, uint32_t PreemptPriority, uint32_t SubPriority)
{
  uint32_t PriorityGroupTmp = (PriorityGroup & (uint32_t)0x07UL);
  uint32_t PreemptPriorityBits;
  uint32_t SubPriorityBits;

  PreemptPriorityBits = ((7UL - PriorityGroupTmp) > (uint32_t)(3U)) ? (uint32_t)(3U) : (uint32_t)(7UL - PriorityGroupTmp);
  SubPriorityBits = ((PriorityGroupTmp + (uint32_t)(3U)) < (uint32_t)7UL) ? (uint32_t)0UL : (uint32_t)((PriorityGroupTmp - 7UL) + (uint32_t)(3U));

  return (
           ((PreemptPriority & (uint32_t)((1UL << (PreemptPriorityBits)) - 1UL)) << SubPriorityBits) |
           ((SubPriority & (uint32_t)((1UL << (SubPriorityBits )) - 1UL)))
         );
}
# 4123 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline void NVIC_DecodePriority (uint32_t Priority, uint32_t PriorityGroup, uint32_t* const pPreemptPriority, uint32_t* const pSubPriority)
{
  uint32_t PriorityGroupTmp = (PriorityGroup & (uint32_t)0x07UL);
  uint32_t PreemptPriorityBits;
  uint32_t SubPriorityBits;

  PreemptPriorityBits = ((7UL - PriorityGroupTmp) > (uint32_t)(3U)) ? (uint32_t)(3U) : (uint32_t)(7UL - PriorityGroupTmp);
  SubPriorityBits = ((PriorityGroupTmp + (uint32_t)(3U)) < (uint32_t)7UL) ? (uint32_t)0UL : (uint32_t)((PriorityGroupTmp - 7UL) + (uint32_t)(3U));

  *pPreemptPriority = (Priority >> SubPriorityBits) & (uint32_t)((1UL << (PreemptPriorityBits)) - 1UL);
  *pSubPriority = (Priority ) & (uint32_t)((1UL << (SubPriorityBits )) - 1UL);
}
# 4146 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline void __NVIC_SetVector(IRQn_Type IRQn, uint32_t vector)
{
  uint32_t *vectors = (uint32_t *)((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->VTOR;
  vectors[(int32_t)IRQn + 16] = vector;
  __builtin_arm_dsb(0xF);
}
# 4162 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t __NVIC_GetVector(IRQn_Type IRQn)
{
  uint32_t *vectors = (uint32_t *)((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->VTOR;
  return vectors[(int32_t)IRQn + 16];
}






__attribute__((__noreturn__)) static __inline void __NVIC_SystemReset(void)
{
  __builtin_arm_dsb(0xF);

  ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->AIRCR = (uint32_t)((0x5FAUL << 16U) |
                           (((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->AIRCR & (7UL << 8U)) |
                            (1UL << 2U) );
  __builtin_arm_dsb(0xF);

  for(;;)
  {
    __builtin_arm_nop();
  }
}
# 4396 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
# 1 "../../../../Library/CMSIS/Core/Include\\mpu_armv8.h" 1 3
# 29 "../../../../Library/CMSIS/Core/Include\\mpu_armv8.h" 3
# 122 "../../../../Library/CMSIS/Core/Include\\mpu_armv8.h" 3
typedef struct {
  uint32_t RBAR;
  uint32_t RLAR;
} ARM_MPU_Region_t;




static __inline void ARM_MPU_Enable(uint32_t MPU_Control)
{
  __builtin_arm_dmb(0xF);
  ((MPU_Type *) ((0xE000E000UL) + 0x0D90UL) )->CTRL = MPU_Control | (1UL );

  ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->SHCSR |= (1UL << 16U);

  __builtin_arm_dsb(0xF);
  __builtin_arm_isb(0xF);
}



static __inline void ARM_MPU_Disable(void)
{
  __builtin_arm_dmb(0xF);

  ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->SHCSR &= ~(1UL << 16U);

  ((MPU_Type *) ((0xE000E000UL) + 0x0D90UL) )->CTRL &= ~(1UL );
  __builtin_arm_dsb(0xF);
  __builtin_arm_isb(0xF);
}
# 188 "../../../../Library/CMSIS/Core/Include\\mpu_armv8.h" 3
static __inline void ARM_MPU_SetMemAttrEx(MPU_Type* mpu, uint8_t idx, uint8_t attr)
{
  const uint8_t reg = idx / 4U;
  const uint32_t pos = ((idx % 4U) * 8U);
  const uint32_t mask = 0xFFU << pos;

  if (reg >= (sizeof(mpu->MAIR) / sizeof(mpu->MAIR[0]))) {
    return;
  }

  mpu->MAIR[reg] = ((mpu->MAIR[reg] & ~mask) | ((attr << pos) & mask));
}





static __inline void ARM_MPU_SetMemAttr(uint8_t idx, uint8_t attr)
{
  ARM_MPU_SetMemAttrEx(((MPU_Type *) ((0xE000E000UL) + 0x0D90UL) ), idx, attr);
}
# 225 "../../../../Library/CMSIS/Core/Include\\mpu_armv8.h" 3
static __inline void ARM_MPU_ClrRegionEx(MPU_Type* mpu, uint32_t rnr)
{
  mpu->RNR = rnr;
  mpu->RLAR = 0U;
}




static __inline void ARM_MPU_ClrRegion(uint32_t rnr)
{
  ARM_MPU_ClrRegionEx(((MPU_Type *) ((0xE000E000UL) + 0x0D90UL) ), rnr);
}
# 255 "../../../../Library/CMSIS/Core/Include\\mpu_armv8.h" 3
static __inline void ARM_MPU_SetRegionEx(MPU_Type* mpu, uint32_t rnr, uint32_t rbar, uint32_t rlar)
{
  mpu->RNR = rnr;
  mpu->RBAR = rbar;
  mpu->RLAR = rlar;
}






static __inline void ARM_MPU_SetRegion(uint32_t rnr, uint32_t rbar, uint32_t rlar)
{
  ARM_MPU_SetRegionEx(((MPU_Type *) ((0xE000E000UL) + 0x0D90UL) ), rnr, rbar, rlar);
}
# 289 "../../../../Library/CMSIS/Core/Include\\mpu_armv8.h" 3
static __inline void ARM_MPU_OrderedMemcpy(volatile uint32_t* dst, const uint32_t* __restrict src, uint32_t len)
{
  uint32_t i;
  for (i = 0U; i < len; ++i)
  {
    dst[i] = src[i];
  }
}







static __inline void ARM_MPU_LoadEx(MPU_Type* mpu, uint32_t rnr, ARM_MPU_Region_t const* table, uint32_t cnt)
{
  const uint32_t rowWordSize = sizeof(ARM_MPU_Region_t)/4U;
  if (cnt == 1U) {
    mpu->RNR = rnr;
    ARM_MPU_OrderedMemcpy(&(mpu->RBAR), &(table->RBAR), rowWordSize);
  } else {
    uint32_t rnrBase = rnr & ~(4U -1U);
    uint32_t rnrOffset = rnr % 4U;

    mpu->RNR = rnrBase;
    while ((rnrOffset + cnt) > 4U) {
      uint32_t c = 4U - rnrOffset;
      ARM_MPU_OrderedMemcpy(&(mpu->RBAR)+(rnrOffset*2U), &(table->RBAR), c*rowWordSize);
      table += c;
      cnt -= c;
      rnrOffset = 0U;
      rnrBase += 4U;
      mpu->RNR = rnrBase;
    }

    ARM_MPU_OrderedMemcpy(&(mpu->RBAR)+(rnrOffset*2U), &(table->RBAR), cnt*rowWordSize);
  }
}






static __inline void ARM_MPU_Load(uint32_t rnr, ARM_MPU_Region_t const* table, uint32_t cnt)
{
  ARM_MPU_LoadEx(((MPU_Type *) ((0xE000E000UL) + 0x0D90UL) ), rnr, table, cnt);
}
# 4397 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 2 3







# 1 "../../../../Library/CMSIS/Core/Include\\pmu_armv8.h" 1 3
# 29 "../../../../Library/CMSIS/Core/Include\\pmu_armv8.h" 3
# 173 "../../../../Library/CMSIS/Core/Include\\pmu_armv8.h" 3
static __inline void ARM_PMU_Enable(void);
static __inline void ARM_PMU_Disable(void);

static __inline void ARM_PMU_Set_EVTYPER(uint32_t num, uint32_t type);

static __inline void ARM_PMU_CYCCNT_Reset(void);
static __inline void ARM_PMU_EVCNTR_ALL_Reset(void);

static __inline void ARM_PMU_CNTR_Enable(uint32_t mask);
static __inline void ARM_PMU_CNTR_Disable(uint32_t mask);

static __inline uint32_t ARM_PMU_Get_CCNTR(void);
static __inline uint32_t ARM_PMU_Get_EVCNTR(uint32_t num);

static __inline uint32_t ARM_PMU_Get_CNTR_OVS(void);
static __inline void ARM_PMU_Set_CNTR_OVS(uint32_t mask);

static __inline void ARM_PMU_Set_CNTR_IRQ_Enable(uint32_t mask);
static __inline void ARM_PMU_Set_CNTR_IRQ_Disable(uint32_t mask);

static __inline void ARM_PMU_CNTR_Increment(uint32_t mask);




static __inline void ARM_PMU_Enable(void)
{
  ((PMU_Type *) (0xE0003000UL) )->CTRL |= (1UL );
}




static __inline void ARM_PMU_Disable(void)
{
  ((PMU_Type *) (0xE0003000UL) )->CTRL &= ~(1UL );
}






static __inline void ARM_PMU_Set_EVTYPER(uint32_t num, uint32_t type)
{
  ((PMU_Type *) (0xE0003000UL) )->EVTYPER[num] = type;
}




static __inline void ARM_PMU_CYCCNT_Reset(void)
{
  ((PMU_Type *) (0xE0003000UL) )->CTRL |= (1UL << 2U);
}




static __inline void ARM_PMU_EVCNTR_ALL_Reset(void)
{
  ((PMU_Type *) (0xE0003000UL) )->CTRL |= (1UL << 1U);
}
# 244 "../../../../Library/CMSIS/Core/Include\\pmu_armv8.h" 3
static __inline void ARM_PMU_CNTR_Enable(uint32_t mask)
{
  ((PMU_Type *) (0xE0003000UL) )->CNTENSET = mask;
}
# 256 "../../../../Library/CMSIS/Core/Include\\pmu_armv8.h" 3
static __inline void ARM_PMU_CNTR_Disable(uint32_t mask)
{
  ((PMU_Type *) (0xE0003000UL) )->CNTENCLR = mask;
}





static __inline uint32_t ARM_PMU_Get_CCNTR(void)
{
  return ((PMU_Type *) (0xE0003000UL) )->CCNTR;
}






static __inline uint32_t ARM_PMU_Get_EVCNTR(uint32_t num)
{
  return (0xFFFFUL ) & ((PMU_Type *) (0xE0003000UL) )->EVCNTR[num];
}







static __inline uint32_t ARM_PMU_Get_CNTR_OVS(void)
{
  return ((PMU_Type *) (0xE0003000UL) )->OVSSET;
}
# 298 "../../../../Library/CMSIS/Core/Include\\pmu_armv8.h" 3
static __inline void ARM_PMU_Set_CNTR_OVS(uint32_t mask)
{
  ((PMU_Type *) (0xE0003000UL) )->OVSCLR = mask;
}
# 310 "../../../../Library/CMSIS/Core/Include\\pmu_armv8.h" 3
static __inline void ARM_PMU_Set_CNTR_IRQ_Enable(uint32_t mask)
{
  ((PMU_Type *) (0xE0003000UL) )->INTENSET = mask;
}
# 322 "../../../../Library/CMSIS/Core/Include\\pmu_armv8.h" 3
static __inline void ARM_PMU_Set_CNTR_IRQ_Disable(uint32_t mask)
{
  ((PMU_Type *) (0xE0003000UL) )->INTENCLR = mask;
}






static __inline void ARM_PMU_CNTR_Increment(uint32_t mask)
{
  ((PMU_Type *) (0xE0003000UL) )->SWINC = mask;
}
# 4405 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 2 3
# 4451 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t SCB_GetFPUType(void)
{
  uint32_t mvfr0;

  mvfr0 = ((FPU_Type *) ((0xE000E000UL) + 0x0F30UL) )->MVFR0;
  if ((mvfr0 & ((0xFUL << 4U) | (0xFUL << 8U))) == 0x220U)
  {
    return 2U;
  }
  else if ((mvfr0 & ((0xFUL << 4U) | (0xFUL << 8U))) == 0x020U)
  {
    return 1U;
  }
  else
  {
    return 0U;
  }
}
# 4489 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t SCB_GetMVEType(void)
{
  const uint32_t mvfr1 = ((FPU_Type *) ((0xE000E000UL) + 0x0F30UL) )->MVFR1;
  if ((mvfr1 & (0xFUL << 8U)) == (0x2U << 8U))
  {
    return 2U;
  }
  else if ((mvfr1 & (0xFUL << 8U)) == (0x1U << 8U))
  {
    return 1U;
  }
  else
  {
    return 0U;
  }
}
# 4514 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
# 1 "../../../../Library/CMSIS/Core/Include\\cachel1_armv7.h" 1 3
# 29 "../../../../Library/CMSIS/Core/Include\\cachel1_armv7.h" 3
# 57 "../../../../Library/CMSIS/Core/Include\\cachel1_armv7.h" 3
__attribute__((always_inline)) static __inline void SCB_EnableICache (void)
{

    if (((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCR & (1UL << 17U)) return;

    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);
    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->ICIALLU = 0UL;
    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);
    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCR |= (uint32_t)(1UL << 17U);
    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);

}






__attribute__((always_inline)) static __inline void SCB_DisableICache (void)
{

    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);
    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCR &= ~(uint32_t)(1UL << 17U);
    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->ICIALLU = 0UL;
    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);

}






__attribute__((always_inline)) static __inline void SCB_InvalidateICache (void)
{

    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);
    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->ICIALLU = 0UL;
    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);

}
# 115 "../../../../Library/CMSIS/Core/Include\\cachel1_armv7.h" 3
__attribute__((always_inline)) static __inline void SCB_InvalidateICache_by_Addr (volatile void *addr, int32_t isize)
{

    if ( isize > 0 ) {
       int32_t op_size = isize + (((uint32_t)addr) & (32U - 1U));
      uint32_t op_addr = (uint32_t)addr ;

      __builtin_arm_dsb(0xF);

      do {
        ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->ICIMVAU = op_addr;
        op_addr += 32U;
        op_size -= 32U;
      } while ( op_size > 0 );

      __builtin_arm_dsb(0xF);
      __builtin_arm_isb(0xF);
    }

}






__attribute__((always_inline)) static __inline void SCB_EnableDCache (void)
{

    uint32_t ccsidr;
    uint32_t sets;
    uint32_t ways;

    if (((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCR & (1UL << 16U)) return;

    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CSSELR = 0U;
    __builtin_arm_dsb(0xF);

    ccsidr = ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCSIDR;


    sets = (uint32_t)((((ccsidr) & (0x7FFFUL << 13U) ) >> 13U ));
    do {
      ways = (uint32_t)((((ccsidr) & (0x3FFUL << 3U)) >> 3U));
      do {
        ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->DCISW = (((sets << 5U) & (0x1FFUL << 5U)) |
                      ((ways << 30U) & (3UL << 30U)) );



      } while (ways-- != 0U);
    } while(sets-- != 0U);
    __builtin_arm_dsb(0xF);

    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCR |= (uint32_t)(1UL << 16U);

    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);

}






__attribute__((always_inline)) static __inline void SCB_DisableDCache (void)
{

    uint32_t ccsidr;
    uint32_t sets;
    uint32_t ways;

    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CSSELR = 0U;
    __builtin_arm_dsb(0xF);

    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCR &= ~(uint32_t)(1UL << 16U);
    __builtin_arm_dsb(0xF);

    ccsidr = ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCSIDR;


    sets = (uint32_t)((((ccsidr) & (0x7FFFUL << 13U) ) >> 13U ));
    do {
      ways = (uint32_t)((((ccsidr) & (0x3FFUL << 3U)) >> 3U));
      do {
        ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->DCCISW = (((sets << 5U) & (0x1FFUL << 5U)) |
                       ((ways << 30U) & (3UL << 30U)) );



      } while (ways-- != 0U);
    } while(sets-- != 0U);

    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);

}






__attribute__((always_inline)) static __inline void SCB_InvalidateDCache (void)
{

    uint32_t ccsidr;
    uint32_t sets;
    uint32_t ways;

    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CSSELR = 0U;
    __builtin_arm_dsb(0xF);

    ccsidr = ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCSIDR;


    sets = (uint32_t)((((ccsidr) & (0x7FFFUL << 13U) ) >> 13U ));
    do {
      ways = (uint32_t)((((ccsidr) & (0x3FFUL << 3U)) >> 3U));
      do {
        ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->DCISW = (((sets << 5U) & (0x1FFUL << 5U)) |
                      ((ways << 30U) & (3UL << 30U)) );



      } while (ways-- != 0U);
    } while(sets-- != 0U);

    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);

}






__attribute__((always_inline)) static __inline void SCB_CleanDCache (void)
{

    uint32_t ccsidr;
    uint32_t sets;
    uint32_t ways;

    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CSSELR = 0U;
    __builtin_arm_dsb(0xF);

    ccsidr = ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCSIDR;


    sets = (uint32_t)((((ccsidr) & (0x7FFFUL << 13U) ) >> 13U ));
    do {
      ways = (uint32_t)((((ccsidr) & (0x3FFUL << 3U)) >> 3U));
      do {
        ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->DCCSW = (((sets << 5U) & (0x1FFUL << 5U)) |
                      ((ways << 30U) & (3UL << 30U)) );



      } while (ways-- != 0U);
    } while(sets-- != 0U);

    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);

}






__attribute__((always_inline)) static __inline void SCB_CleanInvalidateDCache (void)
{

    uint32_t ccsidr;
    uint32_t sets;
    uint32_t ways;

    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CSSELR = 0U;
    __builtin_arm_dsb(0xF);

    ccsidr = ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->CCSIDR;


    sets = (uint32_t)((((ccsidr) & (0x7FFFUL << 13U) ) >> 13U ));
    do {
      ways = (uint32_t)((((ccsidr) & (0x3FFUL << 3U)) >> 3U));
      do {
        ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->DCCISW = (((sets << 5U) & (0x1FFUL << 5U)) |
                       ((ways << 30U) & (3UL << 30U)) );



      } while (ways-- != 0U);
    } while(sets-- != 0U);

    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);

}
# 328 "../../../../Library/CMSIS/Core/Include\\cachel1_armv7.h" 3
__attribute__((always_inline)) static __inline void SCB_InvalidateDCache_by_Addr (volatile void *addr, int32_t dsize)
{

    if ( dsize > 0 ) {
       int32_t op_size = dsize + (((uint32_t)addr) & (32U - 1U));
      uint32_t op_addr = (uint32_t)addr ;

      __builtin_arm_dsb(0xF);

      do {
        ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->DCIMVAC = op_addr;
        op_addr += 32U;
        op_size -= 32U;
      } while ( op_size > 0 );

      __builtin_arm_dsb(0xF);
      __builtin_arm_isb(0xF);
    }

}
# 358 "../../../../Library/CMSIS/Core/Include\\cachel1_armv7.h" 3
__attribute__((always_inline)) static __inline void SCB_CleanDCache_by_Addr (volatile void *addr, int32_t dsize)
{

    if ( dsize > 0 ) {
       int32_t op_size = dsize + (((uint32_t)addr) & (32U - 1U));
      uint32_t op_addr = (uint32_t)addr ;

      __builtin_arm_dsb(0xF);

      do {
        ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->DCCMVAC = op_addr;
        op_addr += 32U;
        op_size -= 32U;
      } while ( op_size > 0 );

      __builtin_arm_dsb(0xF);
      __builtin_arm_isb(0xF);
    }

}
# 388 "../../../../Library/CMSIS/Core/Include\\cachel1_armv7.h" 3
__attribute__((always_inline)) static __inline void SCB_CleanInvalidateDCache_by_Addr (volatile void *addr, int32_t dsize)
{

    if ( dsize > 0 ) {
       int32_t op_size = dsize + (((uint32_t)addr) & (32U - 1U));
      uint32_t op_addr = (uint32_t)addr ;

      __builtin_arm_dsb(0xF);

      do {
        ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->DCCIMVAC = op_addr;
        op_addr += 32U;
        op_size -= 32U;
      } while ( op_size > 0 );

      __builtin_arm_dsb(0xF);
      __builtin_arm_isb(0xF);
    }

}
# 4515 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 2 3
# 4569 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline void DCB_SetAuthCtrl(uint32_t value)
{
    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);
    ((DCB_Type *) (0xE000EDF0UL) )->DAUTHCTRL = value;
    __builtin_arm_dsb(0xF);
    __builtin_arm_isb(0xF);
}







static __inline uint32_t DCB_GetAuthCtrl(void)
{
    return (((DCB_Type *) (0xE000EDF0UL) )->DAUTHCTRL);
}
# 4636 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t DIB_GetAuthStatus(void)
{
    return (((DIB_Type *) (0xE000EFB0UL) )->DAUTHSTATUS);
}
# 4680 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t SysTick_Config(uint32_t ticks)
{
  if ((ticks - 1UL) > (0xFFFFFFUL ))
  {
    return (1UL);
  }

  ((SysTick_Type *) ((0xE000E000UL) + 0x0010UL) )->LOAD = (uint32_t)(ticks - 1UL);
  __NVIC_SetPriority (SysTick_IRQn, (1UL << 3U) - 1UL);
  ((SysTick_Type *) ((0xE000E000UL) + 0x0010UL) )->VAL = 0UL;
  ((SysTick_Type *) ((0xE000E000UL) + 0x0010UL) )->CTRL = (1UL << 2U) |
                   (1UL << 1U) |
                   (1UL );
  return (0UL);
}
# 4740 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
extern volatile int32_t ITM_RxBuffer;
# 4752 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline uint32_t ITM_SendChar (uint32_t ch)
{
  if (((((ITM_Type *) (0xE0000000UL) )->TCR & (1UL )) != 0UL) &&
      ((((ITM_Type *) (0xE0000000UL) )->TER & 1UL ) != 0UL) )
  {
    while (((ITM_Type *) (0xE0000000UL) )->PORT[0U].u32 == 0UL)
    {
      __builtin_arm_nop();
    }
    ((ITM_Type *) (0xE0000000UL) )->PORT[0U].u8 = (uint8_t)ch;
  }
  return (ch);
}
# 4773 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline int32_t ITM_ReceiveChar (void)
{
  int32_t ch = -1;

  if (ITM_RxBuffer != ((int32_t)0x5AA55AA5U))
  {
    ch = ITM_RxBuffer;
    ITM_RxBuffer = ((int32_t)0x5AA55AA5U);
  }

  return (ch);
}
# 4793 "../../../../Library/CMSIS/Core/Include\\core_cm55.h" 3
static __inline int32_t ITM_CheckChar (void)
{

  if (ITM_RxBuffer == ((int32_t)0x5AA55AA5U))
  {
    return (0);
  }
  else
  {
    return (1);
  }
}
# 328 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\system_M55M1.h" 1
# 91 "../../../../Library/Device/Nuvoton/M55M1/Include\\system_M55M1.h"
extern uint32_t CyclesPerUs;
extern uint32_t SystemCoreClock;
extern uint32_t PllClock;
# 114 "../../../../Library/Device/Nuvoton/M55M1/Include\\system_M55M1.h"
typedef void(*VECTOR_TABLE_Type)(void);




extern uint32_t SystemCoreClock;






extern void SystemInit(void);







extern void SystemCoreClockUpdate(void);





extern void SetDebugUartMFP(void);




extern void SetDebugUartCLK(void);




extern void InitDebugUart(void);




extern int32_t InitPreDefMPURegion(const ARM_MPU_Region_t *psMPURegion, uint32_t u32RegionCnt);




extern int32_t SetupMPC(
    const uint32_t u32MPCBaseAddr,
    const uint32_t u32MemBaseAddr, const uint32_t u32MemByteSize,
    const uint32_t u32MemBaseAddr_S, const uint32_t u32MemByteSize_S,
    const uint32_t u32MemBaseAddr_NS, const uint32_t u32MemByteSize_NS
);
# 329 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2



# 1 "../../../../Library/StdDriver/inc\\partition_M55M1_template.h" 1
# 333 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2



# 1 "..\\mpu_config_M55M1.h" 1
# 342 "..\\mpu_config_M55M1.h"
typedef enum
{
    eMPU_ATTR_DEV_nGnRnE,
    eMPU_ATTR_DEV_nGnRE,
    eMPU_ATTR_DEV_nGRE,
    eMPU_ATTR_DEV_GRE,
    eMPU_ATTR_NON_CACHEABLE,
    eMPU_ATTR_CACHEABLE_WTRA,
    eMPU_ATTR_CACHEABLE_WBWARA,
} E_MPU_ATTR;

typedef enum
{
    eMPU_RBAR_SH,
    eMPU_RBAR_RO,
    eMPU_RBAR_NP,
    eMPU_RBAR_XN,
} E_MPU_RBAR;
# 337 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 349 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h"
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\acmp_reg.h" 1
# 33 "../../../../Library/Device/Nuvoton/M55M1/Include\\acmp_reg.h"
typedef struct
{
# 233 "../../../../Library/Device/Nuvoton/M55M1/Include\\acmp_reg.h"
    volatile uint32_t CTL[2];
    volatile uint32_t STATUS;
    volatile uint32_t VREF;
    volatile uint32_t CALCTL;
    volatile uint32_t CALSR;
} ACMP_T;
# 350 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\awf_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\awf_reg.h"
typedef struct
{
# 176 "../../../../Library/Device/Nuvoton/M55M1/Include\\awf_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t HTH;
    volatile uint32_t LTH;
    volatile uint32_t STATUS;
    volatile uint32_t WBINIT;
    volatile const uint32_t RESERVE0[2];
    volatile uint32_t DAT;
    volatile const uint32_t W0;
    volatile const uint32_t W1;
    volatile const uint32_t W2;
    volatile const uint32_t W3;
    volatile const uint32_t W4;
    volatile const uint32_t W5;
    volatile const uint32_t W6;
    volatile const uint32_t W7;
    volatile const uint32_t ACUVAL;
    volatile const uint32_t RESERVE1[1006];
    volatile const uint32_t VERSION;

} AWF_T;
# 351 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\bpwm_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\bpwm_reg.h"
typedef struct
{
# 46 "../../../../Library/Device/Nuvoton/M55M1/Include\\bpwm_reg.h"
    volatile uint32_t RCAPDAT;
    volatile uint32_t FCAPDAT;
} BCAPDAT_T;

typedef struct
{
# 1086 "../../../../Library/Device/Nuvoton/M55M1/Include\\bpwm_reg.h"
    volatile uint32_t CTL0;
    volatile uint32_t CTL1;

    volatile const uint32_t RESERVE0[2];

    volatile uint32_t CLKSRC;
    volatile uint32_t CLKPSC;

    volatile const uint32_t RESERVE1[2];

    volatile uint32_t CNTEN;
    volatile uint32_t CNTCLR;

    volatile const uint32_t RESERVE2[2];

    volatile uint32_t PERIOD;

    volatile const uint32_t RESERVE3[7];

    volatile uint32_t CMPDAT[6];

    volatile const uint32_t RESERVE4[10];

    volatile const uint32_t CNT;

    volatile const uint32_t RESERVE5[7];

    volatile uint32_t WGCTL0;
    volatile uint32_t WGCTL1;
    volatile uint32_t MSKEN;
    volatile uint32_t MSK;

    volatile const uint32_t RESERVE6[5];

    volatile uint32_t POLCTL;
    volatile uint32_t POEN;

    volatile const uint32_t RESERVE7[1];

    volatile uint32_t INTEN;

    volatile const uint32_t RESERVE8[1];

    volatile uint32_t INTSTS;

    volatile const uint32_t RESERVE9[3];

    volatile uint32_t EADCTS0;
    volatile uint32_t EADCTS1;

    volatile const uint32_t RESERVE10[4];

    volatile uint32_t SSCTL;
    volatile uint32_t SSTRG;

    volatile const uint32_t RESERVE11[2];

    volatile uint32_t STATUS;

    volatile const uint32_t RESERVE12[55];

    volatile uint32_t CAPINEN;
    volatile uint32_t CAPCTL;
    volatile const uint32_t CAPSTS;
    BCAPDAT_T CAPDAT[6];

    volatile const uint32_t RESERVE13[5];

    volatile uint32_t CAPIEN;
    volatile uint32_t CAPIF;

    volatile const uint32_t RESERVE14[43];

    volatile const uint32_t PBUF;

    volatile const uint32_t RESERVE15[5];

    volatile const uint32_t CMPBUF[6];
} BPWM_T;
# 352 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\canfd_reg.h" 1
# 32 "../../../../Library/Device/Nuvoton/M55M1/Include\\canfd_reg.h"
typedef struct
{
# 1029 "../../../../Library/Device/Nuvoton/M55M1/Include\\canfd_reg.h"
    volatile const uint32_t RESERVE0[3];

    volatile uint32_t DBTP;
    volatile uint32_t TEST;
    volatile uint32_t RWD;
    volatile uint32_t CCCR;
    volatile uint32_t NBTP;
    volatile uint32_t TSCC;
    volatile uint32_t TSCV;
    volatile uint32_t TOCC;
    volatile uint32_t TOCV;

    volatile const uint32_t RESERVE1[4];

    volatile const uint32_t ECR;
    volatile const uint32_t PSR;
    volatile uint32_t TDCR;

    volatile const uint32_t RESERVE2[1];

    volatile uint32_t IR;
    volatile uint32_t IE;
    volatile uint32_t ILS;
    volatile uint32_t ILE;

    volatile const uint32_t RESERVE3[8];

    volatile uint32_t GFC;
    volatile uint32_t SIDFC;
    volatile uint32_t XIDFC;

    volatile const uint32_t RESERVE4[1];

    volatile uint32_t XIDAM;
    volatile const uint32_t HPMS;
    volatile uint32_t NDAT1;
    volatile uint32_t NDAT2;
    volatile uint32_t RXF0C;
    volatile uint32_t RXF0S;
    volatile uint32_t RXF0A;
    volatile uint32_t RXBC;
    volatile uint32_t RXF1C;
    volatile uint32_t RXF1S;
    volatile uint32_t RXF1A;
    volatile uint32_t RXESC;
    volatile uint32_t TXBC;
    volatile uint32_t TXFQS;
    volatile uint32_t TXESC;
    volatile uint32_t TXBRP;
    volatile uint32_t TXBAR;
    volatile uint32_t TXBCR;
    volatile uint32_t TXBTO;
    volatile uint32_t TXBCF;
    volatile uint32_t TXBTIE;
    volatile uint32_t TXBCIE;

    volatile const uint32_t RESERVE5[2];

    volatile uint32_t TXEFC;
    volatile uint32_t TXEFS;
    volatile uint32_t TXEFA;

} CANFD_T;
# 353 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\ccap_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\ccap_reg.h"
typedef struct
{
# 427 "../../../../Library/Device/Nuvoton/M55M1/Include\\ccap_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t PAR;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile uint32_t PARM;
    volatile const uint32_t RESERVE0[3];
    volatile uint32_t CWSP;
    volatile uint32_t CWS;
    volatile uint32_t PKTSL;
    volatile const uint32_t RESERVE1[1];
    volatile uint32_t FRCTL;
    volatile uint32_t STRIDE;
    volatile const uint32_t RESERVE2[1];
    volatile uint32_t FIFOTH;
    volatile uint32_t CMPADDR;
    volatile uint32_t LUMA_Y1_THD;
    volatile uint32_t PKTSM;
    volatile const uint32_t RESERVE3[1];
    volatile const uint32_t CURADDRP;
    volatile const uint32_t RESERVE4[3];
    volatile uint32_t PKTBA0;
    volatile const uint32_t RESERVE5[39];
    volatile uint32_t MDCTL;
    volatile uint32_t MDTRG_WK;
    volatile const uint32_t RESERVE6[2];
    volatile uint32_t MDTTH;
    volatile const uint32_t MDTSAD;
    volatile const uint32_t RESERVE7[2];
    volatile uint32_t MDWOCTH;
    volatile const uint32_t MDWOC;
    volatile const uint32_t RESERVE8[6];
    volatile uint32_t MDWTH[16];
    volatile const uint32_t MDWSAD[16];
} CCAP_T;
# 354 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\clk_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\clk_reg.h"
typedef struct
{
# 2316 "../../../../Library/Device/Nuvoton/M55M1/Include\\clk_reg.h"
    volatile uint32_t SRCCTL;
    volatile const uint32_t STATUS;
    volatile uint32_t MIRCCTL;
    volatile uint32_t HIRCCTL;
    volatile uint32_t HXTCTL;
    volatile uint32_t HIRC48MCTL;
    volatile uint32_t APLL0CTL;
    volatile uint32_t APLL0SEL;
    volatile uint32_t APLL1CTL;
    volatile uint32_t APLL1SEL;
    volatile uint32_t CLKOCTL;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t CLKDCTL;
    volatile uint32_t CLKDSTS;
    volatile uint32_t CDUPB;
    volatile uint32_t CDLOWB;
    volatile uint32_t STOPREQ;
    volatile const uint32_t STOPACK;
    volatile const uint32_t RESERVE1[110];
    volatile uint32_t ACMPCTL;
    volatile uint32_t AWFCTL;
    volatile uint32_t BPWMCTL;
    volatile uint32_t CANFDCTL;
    volatile uint32_t CCAPCTL;
    volatile uint32_t CRCCTL;
    volatile uint32_t CRYPTOCTL;
    volatile uint32_t DACCTL;
    volatile uint32_t DMICCTL;
    volatile uint32_t EADCCTL;
    volatile uint32_t EBICTL;
    volatile uint32_t ECAPCTL;
    volatile uint32_t EMACCTL;
    volatile uint32_t EPWMCTL;
    volatile uint32_t EQEICTL;
    volatile uint32_t FMCCTL;
    volatile uint32_t GDMACTL;
    volatile uint32_t GPIOCTL;
    volatile uint32_t HSOTGCTL;
    volatile uint32_t HSUSBDCTL;
    volatile uint32_t HSUSBHCTL;
    volatile uint32_t I2CCTL;
    volatile uint32_t I2SCTL;
    volatile uint32_t I3CCTL;
    volatile uint32_t KDFCTL;
    volatile uint32_t KPICTL;
    volatile uint32_t KSCTL;
    volatile uint32_t LPADCCTL;
    volatile uint32_t LPPDMACTL;
    volatile uint32_t LPGPIOCTL;
    volatile uint32_t LPI2CCTL;
    volatile uint32_t LPSPICTL;
    volatile uint32_t LPSRAMCTL;
    volatile uint32_t LPTMRCTL;
    volatile uint32_t LPUARTCTL;
    volatile uint32_t NPUCTL;
    volatile const uint32_t RESERVE2[1];
    volatile uint32_t OTFCCTL;
    volatile uint32_t OTGCTL;
    volatile uint32_t PDMACTL;
    volatile uint32_t PSIOCTL;
    volatile uint32_t QSPICTL;
    volatile uint32_t RTCCTL;
    volatile uint32_t SCCTL;
    volatile uint32_t SCUCTL;
    volatile uint32_t SDHCTL;
    volatile uint32_t SPICTL;
    volatile uint32_t SPIMCTL;
    volatile uint32_t SRAMCTL;
    volatile const uint32_t RESERVE3[2];
    volatile uint32_t STCTL;
    volatile uint32_t TAMPERCTL;
    volatile uint32_t TMRCTL;
    volatile uint32_t TRNGCTL;
    volatile uint32_t TTMRCTL;
    volatile uint32_t UARTCTL;
    volatile uint32_t USBDCTL;
    volatile uint32_t USBHCTL;
    volatile uint32_t USCICTL;
    volatile uint32_t UTCPDCTL;
    volatile uint32_t WDTCTL;
    volatile uint32_t WWDTCTL;
    volatile const uint32_t RESERVE4[65];
    volatile uint32_t SCLKSEL;
    volatile uint32_t BPWMSEL;
    volatile uint32_t CANFDSEL;
    volatile uint32_t CCAPSEL;
    volatile uint32_t CLKOSEL;
    volatile uint32_t DMICSEL;
    volatile uint32_t EADCSEL;
    volatile uint32_t EPWMSEL;
    volatile uint32_t FMCSEL;
    volatile uint32_t I2SSEL;
    volatile uint32_t I3CSEL;
    volatile uint32_t KPISEL;
    volatile uint32_t LPADCSEL;
    volatile uint32_t LPSPISEL;
    volatile uint32_t LPTMRSEL;
    volatile uint32_t LPUARTSEL;
    volatile uint32_t PSIOSEL;
    volatile uint32_t QSPISEL;
    volatile uint32_t SCSEL;
    volatile uint32_t SDHSEL;
    volatile uint32_t SPISEL;
    volatile uint32_t STSEL;
    volatile uint32_t TMRSEL;
    volatile uint32_t TTMRSEL;
    volatile uint32_t UARTSEL0;
    volatile uint32_t UARTSEL1;
    volatile uint32_t USBSEL;
    volatile uint32_t WDTSEL;
    volatile uint32_t WWDTSEL;
    volatile uint32_t DLLSEL;
    volatile const uint32_t RESERVE5[34];
    volatile uint32_t SCLKDIV;
    volatile const uint32_t RESERVE6[1];
    volatile uint32_t HCLKDIV;
    volatile uint32_t PCLKDIV;
    volatile uint32_t CANFDDIV;
    volatile uint32_t DMICDIV;
    volatile uint32_t EADCDIV;
    volatile uint32_t I2SDIV;
    volatile uint32_t KPIDIV;
    volatile uint32_t LPADCDIV;
    volatile uint32_t LPUARTDIV;
    volatile uint32_t PSIODIV;
    volatile uint32_t SCDIV;
    volatile uint32_t SDHDIV;
    volatile uint32_t STDIV;
    volatile uint32_t UARTDIV0;
    volatile uint32_t UARTDIV1;
    volatile uint32_t USBDIV;
    volatile uint32_t VSENSEDIV;

} CLK_T;
# 355 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\crc_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\crc_reg.h"
typedef struct
{
# 163 "../../../../Library/Device/Nuvoton/M55M1/Include\\crc_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t DAT;
    volatile uint32_t SEED;
    volatile const uint32_t CHECKSUM;
    volatile uint32_t POLYNOMIAL;
    volatile const uint32_t RESERVE0[11];
    volatile uint32_t DMACTL;
    volatile uint32_t DMASTS;
    volatile uint32_t SADDR;
    volatile uint32_t DMACNT;
    volatile const uint32_t RESERVE1[1003];
    volatile const uint32_t VERSION;

} CRC_T;
# 356 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\crypto_reg.h" 1
# 25 "../../../../Library/Device/Nuvoton/M55M1/Include\\crypto_reg.h"
typedef struct
{
# 1613 "../../../../Library/Device/Nuvoton/M55M1/Include\\crypto_reg.h"
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile uint32_t PRNG_CTL;
    volatile uint32_t PRNG_SEED;
    volatile const uint32_t PRNG_KEY[8];
    volatile const uint32_t PRNG_STS;
    volatile const uint32_t RESERVE0[7];
    volatile const uint32_t AES_FDBCK[4];
    volatile const uint32_t RESERVE1[8];
    volatile uint32_t AES_GCM_IVCNT[2];
    volatile uint32_t AES_GCM_ACNT[2];
    volatile uint32_t AES_GCM_PCNT[2];
    volatile const uint32_t RESERVE2[2];
    volatile uint32_t AES_FBADDR;
    volatile const uint32_t RESERVE3[23];
    volatile uint32_t AES_CTL;
    volatile const uint32_t AES_STS;
    volatile uint32_t AES_DATIN;
    volatile const uint32_t AES_DATOUT;
    volatile uint32_t AES_KEY[8];
    volatile uint32_t AES_IV[4];
    volatile uint32_t AES_SADDR;
    volatile uint32_t AES_DADDR;
    volatile uint32_t AES_CNT;
    volatile const uint32_t RESERVE4[109];
    volatile uint32_t HMAC_CTL;
    volatile const uint32_t HMAC_STS;
    volatile const uint32_t HMAC_DGST[16];
    volatile uint32_t HMAC_KEYCNT;
    volatile uint32_t HMAC_SADDR;
    volatile uint32_t HMAC_DMACNT;
    volatile uint32_t HMAC_DATIN;
    volatile uint32_t HMAC_FDBCK[88];
    volatile const uint32_t RESERVE5[16];
    volatile uint32_t HMAC_SHA512T;
    volatile uint32_t HMAC_FBADDR;
    volatile const uint32_t HMAC_SHAKEDGST[42];
    volatile const uint32_t RESERVE6[150];
    volatile uint32_t ECC_CTL;
    volatile const uint32_t ECC_STS;
    volatile uint32_t ECC_X1[18];
    volatile uint32_t ECC_Y1[18];
    volatile uint32_t ECC_X2[18];
    volatile uint32_t ECC_Y2[18];
    volatile uint32_t ECC_A[18];
    volatile uint32_t ECC_B[18];
    volatile uint32_t ECC_N[18];
    volatile uint32_t ECC_K[18];
    volatile uint32_t ECC_SADDR;
    volatile uint32_t ECC_DADDR;
    volatile uint32_t ECC_STARTREG;
    volatile uint32_t ECC_WORDCNT;
    volatile uint32_t ECC_DMA_CTL;
    volatile const uint32_t RESERVE7[41];
    volatile uint32_t RSA_CTL;
    volatile const uint32_t RSA_STS;
    volatile uint32_t RSA_SADDR[5];
    volatile uint32_t RSA_DADDR;
    volatile uint32_t RSA_MADDR[7];
    volatile uint32_t RSA_PKADDR;
    volatile const uint32_t RESERVE8[44];
    volatile const uint32_t RSA_DEBUG;
    volatile const uint32_t RESERVE9[3];
    volatile uint32_t CHAPOLY_CTL;
    volatile const uint32_t CHAPOLY_STS;
    volatile uint32_t CHAPOLY_DATIN;
    volatile const uint32_t CHAPOLY_DATOUT;
    volatile uint32_t CHAPOLY_KEY[8];
    volatile uint32_t CHAPOLY_BLOCKCNT;
    volatile uint32_t CHAPOLY_NONCE[3];
    volatile uint32_t CHAPOLY_SADDR;
    volatile uint32_t CHAPOLY_DADDR;
    volatile uint32_t CHAPOLY_CNT;
    volatile uint32_t CHAPOLY_FBADDR;
    volatile uint32_t CHAPOLY_ACNT[2];
    volatile uint32_t CHAPOLY_PCNT[2];
    volatile const uint32_t RESERVE10[168];
    volatile uint32_t PRNG_KSCTL;
    volatile const uint32_t PRNG_KSSTS;
    volatile const uint32_t RESERVE11[2];
    volatile uint32_t AES_KSCTL;
    volatile const uint32_t RESERVE12[7];
    volatile uint32_t HMAC_KSCTL;
    volatile const uint32_t RESERVE13[3];
    volatile uint32_t ECC_KSCTL;
    volatile const uint32_t ECC_KSSTS;
    volatile uint32_t ECC_KSXY;
    volatile const uint32_t RESERVE14[1];
    volatile uint32_t RSA_KSCTL;
    volatile uint32_t RSA_KSSTS0;
    volatile uint32_t RSA_KSSTS1;
    volatile const uint32_t RESERVE15[1];
    volatile uint32_t CHAPOLY_KSCTL;
    volatile const uint32_t RESERVE16[7];
    volatile uint32_t PAP_CTL;
    volatile const uint32_t RESERVE17[30];
    volatile const uint32_t VERSION;

} CRYPTO_T;
# 357 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\dac_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\dac_reg.h"
typedef struct
{
# 156 "../../../../Library/Device/Nuvoton/M55M1/Include\\dac_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t SWTRG;
    volatile uint32_t DAT;
    volatile const uint32_t DATOUT;
    volatile uint32_t STATUS;
    volatile uint32_t TCTL;
    volatile const uint32_t RESERVE[6];
    volatile uint32_t GRPDAT;

} DAC_T;
# 358 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\dmic_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\dmic_reg.h"
typedef struct
{
# 232 "../../../../Library/Device/Nuvoton/M55M1/Include\\dmic_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t DIV;
    volatile const uint32_t STATUS;
    volatile uint32_t LPPDMACTL;
    volatile const uint32_t FIFO;
    volatile uint32_t GAINCTL0;
    volatile uint32_t GAINCTL1;
    volatile const uint32_t RESERVE0[6];
} DMIC_T;


typedef struct
{
# 359 "../../../../Library/Device/Nuvoton/M55M1/Include\\dmic_reg.h"
    volatile uint32_t SINCCTL;
    volatile uint32_t BIQCTL0;
    volatile uint32_t BIQCTL1;
    volatile uint32_t BIQCTL2;
    volatile uint32_t CTL0;
    volatile uint32_t CTL1;
    volatile uint32_t CTL2;
    volatile uint32_t CTL3;
    volatile const uint32_t STATUS0;
    volatile const uint32_t STATUS1;

} VAD_T;
# 359 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\eadc_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\eadc_reg.h"
typedef struct
{
# 842 "../../../../Library/Device/Nuvoton/M55M1/Include\\eadc_reg.h"
    volatile const uint32_t DAT[19];
    volatile const uint32_t CURDAT;
    volatile uint32_t CTL;
    volatile uint32_t SWTRG;
    volatile uint32_t PENDSTS;
    volatile uint32_t OVSTS;
    volatile uint32_t CTL1;
    volatile const uint32_t RESERVE0[7];
    volatile uint32_t SCTL[19];
    volatile const uint32_t RESERVE1[1];
    volatile uint32_t INTSRC[4];
    volatile uint32_t CMP[4];
    volatile const uint32_t STATUS0;
    volatile const uint32_t STATUS1;
    volatile uint32_t STATUS2;
    volatile const uint32_t STATUS3;
    volatile const uint32_t DDAT[4];
    volatile const uint32_t RESERVE2[1];
    volatile uint32_t CALCTL;
    volatile uint32_t CALSR;
    volatile const uint32_t RESERVE3[5];
    volatile uint32_t PDMACTL;
    volatile const uint32_t RESERVE4[3];
    volatile uint32_t MCTL1[19];
    volatile const uint32_t RESERVE5[29];
    volatile const uint32_t DAT19[9];
    volatile const uint32_t RESERVE6[3];
    volatile uint32_t SCTL19[9];
    volatile const uint32_t RESERVE7[3];
    volatile uint32_t M19CTL1[9];
} EADC_T;
# 360 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\ebi_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\ebi_reg.h"
typedef struct
{
# 242 "../../../../Library/Device/Nuvoton/M55M1/Include\\ebi_reg.h"
    volatile uint32_t CTL0;
    volatile uint32_t TCTL0;
    volatile const uint32_t RESERVE0[2];
    volatile uint32_t CTL1;
    volatile uint32_t TCTL1;
    volatile const uint32_t RESERVE1[2];
    volatile uint32_t CTL2;
    volatile uint32_t TCTL2;

} EBI_T;
# 361 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\ecap_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\ecap_reg.h"
typedef struct
{
# 241 "../../../../Library/Device/Nuvoton/M55M1/Include\\ecap_reg.h"
    volatile uint32_t CNT;
    volatile uint32_t HLD0;
    volatile uint32_t HLD1;
    volatile uint32_t HLD2;
    volatile uint32_t CNTCMP;
    volatile uint32_t CTL0;
    volatile uint32_t CTL1;
    volatile uint32_t STATUS;
} ECAP_T;
# 362 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\epwm_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\epwm_reg.h"
typedef struct
{
# 46 "../../../../Library/Device/Nuvoton/M55M1/Include\\epwm_reg.h"
    volatile uint32_t RCAPDAT;
    volatile uint32_t FCAPDAT;
} ECAPDAT_T;

typedef struct
{
# 2098 "../../../../Library/Device/Nuvoton/M55M1/Include\\epwm_reg.h"
    volatile uint32_t CTL0;
    volatile uint32_t CTL1;
    volatile uint32_t SYNC;
    volatile uint32_t SWSYNC;
    volatile uint32_t CLKSRC;
    volatile const uint32_t RESERVE0_0[3];
    volatile uint32_t CNTEN;
    volatile uint32_t CNTCLR;
    volatile uint32_t LOAD;

    volatile const uint32_t RESERVE0[1];

    volatile uint32_t PERIOD[6];

    volatile const uint32_t RESERVE1[2];

    volatile uint32_t CMPDAT[6];

    volatile const uint32_t RESERVE2[2];

    volatile const uint32_t RESERVE2_1[3];

    volatile const uint32_t RESERVE3[1];

    volatile uint32_t PHS[3];

    volatile const uint32_t RESERVE4[1];

    volatile const uint32_t CNT[6];

    volatile const uint32_t RESERVE5[2];

    volatile uint32_t WGCTL0;
    volatile uint32_t WGCTL1;
    volatile uint32_t MSKEN;
    volatile uint32_t MSK;
    volatile uint32_t BNF;
    volatile uint32_t FAILBRK;
    volatile uint32_t BRKCTL[3];
    volatile uint32_t POLCTL;
    volatile uint32_t POEN;
    volatile uint32_t SWBRK;
    volatile uint32_t INTEN0;
    volatile uint32_t INTEN1;
    volatile uint32_t INTSTS0;
    volatile uint32_t INTSTS1;

    volatile const uint32_t RESERVE6[1];

    volatile uint32_t DACTRGEN;
    volatile uint32_t EADCTS0;
    volatile uint32_t EADCTS1;
    volatile uint32_t FTCMPDAT[3];

    volatile const uint32_t RESERVE7[1];

    volatile uint32_t SSCTL;
    volatile uint32_t SSTRG;
    volatile uint32_t LEBCTL;
    volatile uint32_t LEBCNT;
    volatile uint32_t STATUS;

    volatile const uint32_t RESERVE8[3];

    volatile uint32_t IFA[6];

    volatile const uint32_t RESERVE9[2];

    volatile uint32_t AINTSTS;
    volatile uint32_t AINTEN;
    volatile uint32_t APDMACTL;

    volatile const uint32_t RESERVE10[1];

    volatile uint32_t FDEN;
    volatile uint32_t FDCTL[6];
    volatile uint32_t FDIEN;
    volatile uint32_t FDSTS;
    volatile uint32_t EADCPSCCTL;
    volatile uint32_t EADCPSC0;
    volatile uint32_t EADCPSC1;
    volatile uint32_t EADCPSCNT0;
    volatile uint32_t EADCPSCNT1;

    volatile const uint32_t RESERVE11[26];

    volatile uint32_t CAPINEN;
    volatile uint32_t CAPCTL;
    volatile const uint32_t CAPSTS;
    ECAPDAT_T CAPDAT[6];
    volatile uint32_t PDMACTL;
    volatile const uint32_t PDMACAP[3];

    volatile const uint32_t RESERVE12[1];

    volatile uint32_t CAPIEN;
    volatile uint32_t CAPIF;

    volatile uint32_t CAPNF[6];
    volatile uint32_t EXTETCTL[6];
    volatile uint32_t SWEOFCTL;
    volatile uint32_t SWEOFTRG;
    volatile uint32_t CLKPSC[6];
    volatile uint32_t RDTCNT[3];
    volatile uint32_t FDTCNT[3];
    volatile uint32_t DTCTL;


    volatile const uint32_t RESERVE13[16];

    volatile const uint32_t PBUF[6];
    volatile const uint32_t CMPBUF[6];
    volatile const uint32_t RESERVE14[3];
    volatile const uint32_t FTCBUF[3];
    volatile uint32_t FTCI;
    volatile const uint32_t CPSCBUF[6];
    volatile const uint32_t IFACNT[6];

} EPWM_T;
# 363 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\eqei_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\eqei_reg.h"
typedef struct
{
# 270 "../../../../Library/Device/Nuvoton/M55M1/Include\\eqei_reg.h"
    volatile uint32_t CNT;
    volatile uint32_t CNTHOLD;
    volatile uint32_t CNTLATCH;
    volatile uint32_t CNTCMP;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t CNTMAX;
    volatile uint32_t CTL;
    volatile uint32_t CTL2;
    volatile uint32_t UTCNT;
    volatile uint32_t UTCMP;
    volatile const uint32_t RESERVE1[1];
    volatile uint32_t STATUS;
} EQEI_T;
# 364 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\fmc_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\fmc_reg.h"
typedef struct
{
# 408 "../../../../Library/Device/Nuvoton/M55M1/Include\\fmc_reg.h"
    volatile uint32_t ISPCTL;
    volatile uint32_t ISPADDR;
    volatile uint32_t ISPDAT;
    volatile uint32_t ISPCMD;
    volatile uint32_t ISPTRG;
    volatile const uint32_t RESERVE;
    volatile uint32_t FTCTL;
    volatile uint32_t ICPCTL;

    volatile const uint32_t RESERVE0[8];

    volatile uint32_t ISPSTS;

    volatile const uint32_t RESERVE1[2];

    volatile uint32_t CYCCTL;

    volatile const uint32_t RESERVE2[12];

    volatile uint32_t MPDAT0;
    volatile uint32_t MPDAT1;
    volatile uint32_t MPDAT2;
    volatile uint32_t MPDAT3;

    volatile const uint32_t RESERVE3[12];

    volatile const uint32_t MPSTS;
    volatile const uint32_t MPADDR;

    volatile const uint32_t RESERVE4[2];

    volatile const uint32_t XOMR0STS;
    volatile const uint32_t XOMR1STS;
    volatile const uint32_t XOMR2STS;
    volatile const uint32_t XOMR3STS;
    volatile const uint32_t XOMSTS;
    volatile const uint32_t RESERVE5[11];
    union
    {
        volatile uint32_t APWPROT[2];
        struct
        {
            volatile uint32_t APWPROT0;
            volatile uint32_t APWPROT1;
        };
    };
    volatile uint32_t APWPKEEP;
    volatile uint32_t SCACT;
} FMC_T;
# 365 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\gpio_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\gpio_reg.h"
typedef struct
{
# 278 "../../../../Library/Device/Nuvoton/M55M1/Include\\gpio_reg.h"
    volatile uint32_t MODE;
    volatile uint32_t DINOFF;
    volatile uint32_t DOUT;
    volatile uint32_t DATMSK;
    volatile const uint32_t PIN;
    volatile uint32_t DBEN;
    volatile uint32_t INTTYPE;
    volatile uint32_t INTEN;
    volatile uint32_t INTSRC;
    volatile uint32_t SMTEN;
    volatile uint32_t SLEWCTL;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t PUSEL;
    volatile uint32_t DBCTL;

} GPIO_T;

typedef struct
{
# 670 "../../../../Library/Device/Nuvoton/M55M1/Include\\gpio_reg.h"
    volatile uint32_t INT0_INNF;
    volatile uint32_t INT0_EDETCTL;
    volatile uint32_t INT0_EDINTEN;
    volatile uint32_t INT0_EDSTS;

    volatile const uint32_t RESERVE0[4];

    volatile uint32_t INT1_INNF;
    volatile uint32_t INT1_EDETCTL;
    volatile uint32_t INT1_EDINTEN;
    volatile uint32_t INT1_EDSTS;

    volatile const uint32_t RESERVE1[4];

    volatile uint32_t INT2_INNF;
    volatile uint32_t INT2_EDETCTL;
    volatile uint32_t INT2_EDINTEN;
    volatile uint32_t INT2_EDSTS;

    volatile const uint32_t RESERVE2[4];

    volatile uint32_t INT3_INNF;
    volatile uint32_t INT3_EDETCTL;
    volatile uint32_t INT3_EDINTEN;
    volatile uint32_t INT3_EDSTS;

    volatile const uint32_t RESERVE3[4];

    volatile uint32_t INT4_INNF;
    volatile uint32_t INT4_EDETCTL;
    volatile uint32_t INT4_EDINTEN;
    volatile uint32_t INT4_EDSTS;

    volatile const uint32_t RESERVE4[4];

    volatile uint32_t INT5_INNF;
    volatile uint32_t INT5_EDETCTL;
    volatile uint32_t INT5_EDINTEN;
    volatile uint32_t INT5_EDSTS;

    volatile const uint32_t RESERVE5[4];

    volatile uint32_t INT6_INNF;
    volatile uint32_t INT6_EDETCTL;
    volatile uint32_t INT6_EDINTEN;
    volatile uint32_t INT6_EDSTS;

    volatile const uint32_t RESERVE6[4];

    volatile uint32_t INT7_INNF;
    volatile uint32_t INT7_EDETCTL;
    volatile uint32_t INT7_EDINTEN;
    volatile uint32_t INT7_EDSTS;

} GPIO_INT_T;
# 366 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsotg_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsotg_reg.h"
typedef struct
{
# 250 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsotg_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t PHYCTL;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile const uint32_t STATUS;

} HSOTG_T;
# 367 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsusbd_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsusbd_reg.h"
typedef struct
{
# 280 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsusbd_reg.h"
    union
    {
        volatile uint32_t EPDAT;
        volatile uint8_t EPDAT_BYTE;

    };

    volatile uint32_t EPINTSTS;
    volatile uint32_t EPINTEN;
    volatile const uint32_t EPDATCNT;
    volatile uint32_t EPRSPCTL;
    volatile uint32_t EPMPS;
    volatile uint32_t EPTXCNT;
    volatile uint32_t EPCFG;
    volatile uint32_t EPBUFST;
    volatile uint32_t EPBUFEND;

} HSUSBD_EP_T;

typedef struct
{
# 996 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsusbd_reg.h"
    volatile const uint32_t GINTSTS;

    volatile const uint32_t RESERVE0[1];

    volatile uint32_t GINTEN;

    volatile const uint32_t RESERVE1[1];

    volatile uint32_t BUSINTSTS;
    volatile uint32_t BUSINTEN;
    volatile uint32_t OPER;
    volatile const uint32_t FRAMECNT;
    volatile uint32_t FADDR;
    volatile uint32_t TEST;

    union
    {
        volatile uint32_t CEPDAT;
        volatile uint8_t CEPDAT_BYTE;

    };

    volatile uint32_t CEPCTL;
    volatile uint32_t CEPINTEN;
    volatile uint32_t CEPINTSTS;
    volatile uint32_t CEPTXCNT;
    volatile const uint32_t CEPRXCNT;
    volatile const uint32_t CEPDATCNT;
    volatile const uint32_t SETUP1_0;
    volatile const uint32_t SETUP3_2;
    volatile const uint32_t SETUP5_4;
    volatile const uint32_t SETUP7_6;
    volatile uint32_t CEPBUFST;
    volatile uint32_t CEPBUFEND;
    volatile uint32_t DMACTL;
    volatile uint32_t DMACNT;

    HSUSBD_EP_T EP[18];


    volatile const uint32_t RESERVE2[241];

    volatile uint32_t BCDC;
    volatile uint32_t LPMCSR;
    volatile uint32_t DMAADDR;
    volatile uint32_t PHYCTL;

} HSUSBD_T;
# 368 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsusbh_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsusbh_reg.h"
typedef struct
{
# 419 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsusbh_reg.h"
    volatile const uint32_t EHCVNR;
    volatile const uint32_t EHCSPR;
    volatile const uint32_t EHCCPR;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t UCMDR;
    volatile uint32_t USTSR;
    volatile uint32_t UIENR;
    volatile uint32_t UFINDR;
    volatile const uint32_t RESERVE1[1];
    volatile uint32_t UPFLBAR;
    volatile uint32_t UCALAR;
    volatile const uint32_t RESERVE2[9];
    volatile uint32_t UCFGR;
    volatile uint32_t UPSCR[1];
} HSUSBH_T;

typedef struct
{
# 929 "../../../../Library/Device/Nuvoton/M55M1/Include\\hsusbh_reg.h"
    volatile const uint32_t HcRevision;
    volatile uint32_t HcControl;
    volatile uint32_t HcCommandStatus;
    volatile uint32_t HcInterruptStatus;
    volatile uint32_t HcInterruptEnable;
    volatile uint32_t HcInterruptDisable;
    volatile uint32_t HcHCCA;
    volatile uint32_t HcPeriodCurrentED;
    volatile uint32_t HcControlHeadED;
    volatile uint32_t HcControlCurrentED;
    volatile uint32_t HcBulkHeadED;
    volatile uint32_t HcBulkCurrentED;
    volatile uint32_t HcDoneHead;
    volatile uint32_t HcFmInterval;
    volatile const uint32_t HcFmRemaining;
    volatile const uint32_t HcFmNumber;
    volatile uint32_t HcPeriodicStart;
    volatile uint32_t HcLSThreshold;
    volatile uint32_t HcRhDescriptorA;
    volatile uint32_t HcRhDescriptorB;
    volatile uint32_t HcRhStatus;
    volatile uint32_t HcRhPortStatus[6];

} USBH1_T;
# 369 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\i2c_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\i2c_reg.h"
typedef struct
{
# 515 "../../../../Library/Device/Nuvoton/M55M1/Include\\i2c_reg.h"
    volatile uint32_t CTL0;
    volatile uint32_t ADDR0;
    volatile uint32_t DAT;
    volatile const uint32_t STATUS0;
    volatile uint32_t CLKDIV;
    volatile uint32_t TOCTL;
    volatile uint32_t ADDR1;
    volatile uint32_t ADDR2;
    volatile uint32_t ADDR3;
    volatile uint32_t ADDRMSK0;
    volatile uint32_t ADDRMSK1;
    volatile uint32_t ADDRMSK2;
    volatile uint32_t ADDRMSK3;

    volatile const uint32_t RESERVE0[2];

    volatile uint32_t WKCTL;
    volatile uint32_t WKSTS;
    volatile uint32_t CTL1;
    volatile uint32_t STATUS1;
    volatile uint32_t TMCTL;
    volatile uint32_t BUSCTL;
    volatile uint32_t BUSTCTL;
    volatile uint32_t BUSSTS;
    volatile uint32_t PKTSIZE;
    volatile const uint32_t PKTCRC;
    volatile uint32_t BUSTOUT;
    volatile uint32_t CLKTOUT;

} I2C_T;
# 370 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\i2s_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\i2s_reg.h"
typedef struct
{
# 461 "../../../../Library/Device/Nuvoton/M55M1/Include\\i2s_reg.h"
    volatile uint32_t CTL0;
    volatile uint32_t CLKDIV;
    volatile uint32_t IEN;
    volatile uint32_t STATUS0;
    volatile uint32_t TXFIFO;
    volatile const uint32_t RXFIFO;

    volatile const uint32_t RESERVE0[2];

    volatile uint32_t CTL1;
    volatile uint32_t STATUS1;

} I2S_T;
# 371 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\i3c_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\i3c_reg.h"
typedef struct
{
    volatile uint32_t DEVCTL;
    volatile uint32_t DEVADDR;

    volatile const uint32_t RESERVE0;

    volatile uint32_t CMDQUE;
    volatile const uint32_t RESPQUE;
    volatile uint32_t TXRXDAT;
    union
    {
        volatile uint32_t IBIQUE;
        volatile uint32_t IBISTS;
    };
    volatile uint32_t QUETHCTL;
    volatile uint32_t DBTHCTL;

    volatile const uint32_t RESERVE1[4];

    volatile uint32_t RSTCTL;
    volatile uint32_t SLVEVNTS;
    volatile uint32_t INTSTS;
    volatile uint32_t INTSTSEN;
    volatile uint32_t INTEN;

    volatile const uint32_t RESERVE2;

    volatile uint32_t QUESTSLV;
    volatile const uint32_t DBSTSLV;
    volatile const uint32_t PRESENTS;
    volatile const uint32_t CCCDEVS;
    volatile const uint32_t ADDRTP;
    volatile const uint32_t CHRTP;

    volatile const uint32_t RESERVE3[2];

    volatile const uint32_t VENDOR;
    volatile uint32_t SLVMID;
    volatile uint32_t SLVPID;
    volatile uint32_t SLVCHAR;
    volatile const uint32_t SLVMXLEN;
    volatile const uint32_t MXRDTURN;
    volatile uint32_t MXDS;

    volatile const uint32_t RESERVE4;

    volatile uint32_t SIR;

    volatile const uint32_t RESERVE5;

    volatile uint32_t SIRDAT;
    volatile uint32_t SIRRESP;

    volatile const uint32_t RESERVE6[5];

    volatile uint32_t DEVCTLE;
    volatile uint32_t SCLOD;
    volatile uint32_t SCLPP;
    volatile uint32_t SCLFM;
    volatile uint32_t SCLFMP;

    volatile const uint32_t RESERVE7[3];

    volatile uint32_t SDAHOLD;
    volatile uint32_t BUSFAT;
    volatile uint32_t BUSIDLET;
    volatile const uint32_t SCLLOW;

    volatile const uint32_t RESERVE8[72];

    volatile const uint32_t DEV1CH[4];
    volatile const uint32_t DEV2CH[4];
    volatile const uint32_t DEV3CH[4];
    volatile const uint32_t DEV4CH[4];
    volatile const uint32_t DEV5CH[4];
    volatile const uint32_t DEV6CH[4];
    volatile const uint32_t DEV7CH[4];

    volatile const uint32_t RESERVE9[4];

    volatile uint32_t DEV1ADR;
    volatile uint32_t DEV2ADR;
    volatile uint32_t DEV3ADR;
    volatile uint32_t DEV4ADR;
    volatile uint32_t DEV5ADR;
    volatile uint32_t DEV6ADR;
    volatile uint32_t DEV7ADR;
} I3C_T;
# 372 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\kdf_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\kdf_reg.h"
typedef struct
{
# 192 "../../../../Library/Device/Nuvoton/M55M1/Include\\kdf_reg.h"
    volatile uint32_t CTL;
    volatile const uint32_t STS;
    volatile uint32_t KLEN;
    volatile const uint32_t RESERVE0[13];
    volatile uint32_t KEYIN[8];
    volatile const uint32_t RESERVE1[8];
    volatile const uint32_t KEYOUT[8];
    volatile const uint32_t RESERVE2[24];
    volatile uint32_t SALT[8];
    volatile const uint32_t RESERVE3[56];
    volatile uint32_t LABEL[3];
    volatile const uint32_t RESERVE4[29];
    volatile uint32_t CTXT[4];
    volatile const uint32_t RESERVE5[796];
    volatile uint32_t KSCTL;
    volatile const uint32_t KSSTS;
    volatile uint32_t KSSIZE;
    volatile const uint32_t RESERVE6[60];
    volatile const uint32_t VERSION;

} KDF_T;
# 373 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\kpi_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\kpi_reg.h"
typedef struct
{
# 196 "../../../../Library/Device/Nuvoton/M55M1/Include\\kpi_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t Reserved0;
    volatile uint32_t STATUS;
    volatile uint32_t Reserved1;
    volatile const uint32_t KST[2];
    volatile uint32_t KPF[2];
    volatile uint32_t KRF[2];
    volatile uint32_t DLYCTL;

} KPI_T;
# 374 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\ks_reg.h" 1
# 24 "../../../../Library/Device/Nuvoton/M55M1/Include\\ks_reg.h"
typedef struct
{
# 232 "../../../../Library/Device/Nuvoton/M55M1/Include\\ks_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t METADATA;
    volatile uint32_t STS;
    volatile const uint32_t REMAIN;
    volatile uint32_t SCMBKEY[4];
    volatile uint32_t KEY[8];
    volatile const uint32_t OTPSTS;
    volatile const uint32_t REMKCNT;
    volatile const uint32_t RESERVE0[1005];
    volatile const uint32_t VERSION;

} KS_T;
# 375 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpadc_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpadc_reg.h"
typedef struct
{
# 397 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpadc_reg.h"
    volatile const uint32_t ADDR[32];
    volatile uint32_t ADCR;
    volatile uint32_t ADCHER;
    volatile uint32_t ADCMPR[2];
    volatile uint32_t ADSR0;
    volatile const uint32_t ADSR1;
    volatile const uint32_t ADSR2;
    volatile const uint32_t RESERVE1[1];
    volatile uint32_t ESMPCTL;
    volatile uint32_t CFDCTL;
    volatile const uint32_t RESERVE2[22];
    volatile const uint32_t ADPDMA;
    volatile const uint32_t RESERVE3[31];
    volatile uint32_t ADCAL;
    volatile uint32_t ADCALSTS;
    volatile const uint32_t RESERVE4[414];
    volatile uint32_t AUTOCTL;
    volatile uint32_t AUTOSTRG;
    volatile const uint32_t RESERVE5[2];
    volatile uint32_t AUTOSTS;
    volatile const uint32_t RESERVE6[502];
    volatile uint32_t ALDOCTL;

} LPADC_T;
# 376 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpgpio_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpgpio_reg.h"
typedef struct
{
# 309 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpgpio_reg.h"
    volatile uint32_t MODE;
    volatile uint32_t DOUT;
    volatile const uint32_t PIN;
    volatile uint32_t DSRST;
    volatile uint32_t DRST;
} LPGPIO_T;
# 377 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpi2c_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpi2c_reg.h"
typedef struct
{
# 378 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpi2c_reg.h"
    volatile uint32_t CTL0;
    volatile uint32_t ADDR0;
    volatile uint32_t DAT;
    volatile const uint32_t STATUS0;
    volatile uint32_t CLKDIV;
    volatile uint32_t TOCTL;
    volatile uint32_t ADDR1;
    volatile uint32_t ADDR2;
    volatile uint32_t ADDR3;
    volatile uint32_t ADDRMSK0;
    volatile uint32_t ADDRMSK1;
    volatile uint32_t ADDRMSK2;
    volatile uint32_t ADDRMSK3;
    volatile const uint32_t RESERVE0[2];
    volatile uint32_t WKCTL;
    volatile uint32_t WKSTS;
    volatile uint32_t CTL1;
    volatile uint32_t STATUS1;
    volatile uint32_t TMCTL;
    volatile const uint32_t RESERVE1[8];
    volatile uint32_t AUTOCTL;
    volatile uint32_t AUTOSTS;
    volatile uint32_t AUTOCNT;

} LPI2C_T;
# 378 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\lppdma_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\lppdma_reg.h"
typedef struct
{
# 112 "../../../../Library/Device/Nuvoton/M55M1/Include\\lppdma_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t SA;
    volatile uint32_t DA;
    volatile uint32_t NEXT;
} LPDSCT_T;

typedef struct
{
# 316 "../../../../Library/Device/Nuvoton/M55M1/Include\\lppdma_reg.h"
    LPDSCT_T LPDSCT[4];
    volatile const uint32_t CURSCAT[4];
    volatile const uint32_t RESERVE1[236];
    volatile uint32_t CHCTL;
    volatile uint32_t PAUSE;
    volatile uint32_t SWREQ;
    volatile const uint32_t TRGSTS;
    volatile uint32_t PRISET;
    volatile uint32_t PRICLR;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile uint32_t ABTSTS;
    volatile uint32_t TDSTS;
    volatile uint32_t ALIGN;
    volatile const uint32_t TACTSTS;
    volatile const uint32_t RESERVE2[12];
    volatile uint32_t CHRST;
    volatile const uint32_t RESERVE4[7];
    volatile uint32_t REQSEL0_3;

} LPPDMA_T;
# 379 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpspi_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpspi_reg.h"
typedef struct
{
# 413 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpspi_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t CLKDIV;
    volatile uint32_t SSCTL;
    volatile uint32_t PDMACTL;
    volatile uint32_t FIFOCTL;
    volatile uint32_t STATUS;
    volatile const uint32_t STATUS2;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t TX;
    volatile const uint32_t RESERVE1[3];
    volatile const uint32_t RX;
    volatile const uint32_t RESERVE2[5];
    volatile uint32_t INTERNAL;
    volatile const uint32_t RESERVE3[1];
    volatile uint32_t AUTOCTL;
    volatile uint32_t AUTOSTS;

} LPSPI_T;
# 380 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\lptmr_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\lptmr_reg.h"
typedef struct
{
# 470 "../../../../Library/Device/Nuvoton/M55M1/Include\\lptmr_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t CMP;
    volatile uint32_t INTSTS;
    volatile uint32_t CNT;
    volatile const uint32_t CAP;
    volatile uint32_t EXTCTL;
    volatile uint32_t EINTSTS;
    volatile uint32_t TRGCTL;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t CAPNF;
    volatile const uint32_t RESERVE1[6];
    volatile uint32_t PWMCTL;
    volatile uint32_t PWMCLKPSC;
    volatile uint32_t PWMCNTCLR;
    volatile uint32_t PWMPERIOD;
    volatile uint32_t PWMCMPDAT;
    volatile const uint32_t PWMCNT;
    volatile uint32_t PWMPOLCTL;
    volatile uint32_t PWMPOCTL;
    volatile uint32_t PWMINTEN0;
    volatile uint32_t PWMINTSTS0;
    volatile uint32_t PWMTRGCTL;
    volatile uint32_t PWMSTATUS;
    volatile const uint32_t PWMPBUF;
    volatile const uint32_t PWMCMPBUF;
    volatile const uint32_t RESERVE2[12];
    volatile uint32_t PWMIFA;
    volatile uint32_t PWMAINTSTS;
    volatile uint32_t PWMAINTEN;
    volatile uint32_t PWMAPDMACTL;
} LPTMR_T;
# 381 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpuart_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpuart_reg.h"
typedef struct
{
# 731 "../../../../Library/Device/Nuvoton/M55M1/Include\\lpuart_reg.h"
    volatile uint32_t DAT;
    volatile uint32_t INTEN;
    volatile uint32_t FIFO;
    volatile uint32_t LINE;
    volatile uint32_t MODEM;
    volatile uint32_t MODEMSTS;
    volatile uint32_t FIFOSTS;
    volatile uint32_t INTSTS;
    volatile uint32_t TOUT;
    volatile uint32_t BAUD;
    volatile uint32_t RESERVE0[1];
    volatile uint32_t ALTCTL;
    volatile uint32_t FUNCSEL;
    volatile uint32_t RESERVE1[2];
    volatile uint32_t BRCOMP;
    volatile uint32_t WKCTL;
    volatile uint32_t WKSTS;
    volatile uint32_t DWKCOMP;
    volatile uint32_t RS485DD;
    volatile uint32_t RESERVE2[2];
    volatile uint32_t AUTOCTL;
    volatile uint32_t AUTOSTS;
    volatile const uint32_t RESERVE3[999];
    volatile const uint32_t VERSION;

} LPUART_T;
# 382 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\otfc_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\otfc_reg.h"
typedef struct
{
# 138 "../../../../Library/Device/Nuvoton/M55M1/Include\\otfc_reg.h"
    volatile uint32_t SADDR;
    volatile uint32_t EADDR;
    volatile uint32_t KSCTL;
    volatile const uint32_t RESERVE1[1];
    volatile uint32_t KEY0;
    volatile uint32_t KEY1;
    volatile uint32_t KEY2;
    volatile uint32_t KEY3;
    volatile const uint32_t RESERVE2[4];
    volatile uint32_t SCRAMBLE;
    volatile uint32_t NONCE0;
    volatile uint32_t NONCE1;
    volatile uint32_t NONCE2;
} OTFC_PR_T;

typedef struct
{
# 275 "../../../../Library/Device/Nuvoton/M55M1/Include\\otfc_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t STS;
    volatile const uint32_t RESERVE0[2];

    OTFC_PR_T PR[4];
} OTFC_T;
# 383 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\otg_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\otg_reg.h"
typedef struct
{
# 249 "../../../../Library/Device/Nuvoton/M55M1/Include\\otg_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t PHYCTL;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile const uint32_t STATUS;

} OTG_T;
# 384 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\pdma_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\pdma_reg.h"
typedef struct
{
# 115 "../../../../Library/Device/Nuvoton/M55M1/Include\\pdma_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t SA;
    volatile uint32_t DA;
    volatile uint32_t NEXT;
} DSCT_T;

typedef struct
{
# 790 "../../../../Library/Device/Nuvoton/M55M1/Include\\pdma_reg.h"
    DSCT_T DSCT[16];
    volatile const uint32_t CURSCAT[16];

    volatile const uint32_t RESERVE1[176];

    volatile uint32_t CHCTL;
    volatile uint32_t PAUSE;
    volatile uint32_t SWREQ;
    volatile const uint32_t TRGSTS;
    volatile uint32_t PRISET;
    volatile uint32_t PRICLR;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile uint32_t ABTSTS;
    volatile uint32_t TDSTS;
    volatile uint32_t ALIGN;
    volatile const uint32_t TACTSTS;
    volatile uint32_t TOUTPSC0_7;
    volatile uint32_t TOUTEN;
    volatile uint32_t TOUTIEN;
    volatile const uint32_t RESERVE2[1];
    volatile uint32_t TOC0_1;
    volatile uint32_t TOC2_3;
    volatile uint32_t TOC4_5;
    volatile uint32_t TOC6_7;
    volatile uint32_t TOC8_9;
    volatile uint32_t TOC10_11;
    volatile uint32_t TOC12_13;
    volatile uint32_t TOC14_15;
    volatile uint32_t CHRST;

    volatile const uint32_t RESERVE3[1];

    volatile uint32_t TOUTPSC8_15;

    volatile const uint32_t RESERVE4[5];

    volatile uint32_t REQSEL0_3;
    volatile uint32_t REQSEL4_7;
    volatile uint32_t REQSEL8_11;
    volatile uint32_t REQSEL12_15;
} PDMA_T;
# 385 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\pmc_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\pmc_reg.h"
typedef struct
{
# 885 "../../../../Library/Device/Nuvoton/M55M1/Include\\pmc_reg.h"
    volatile uint32_t PWRCTL;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t PLCTL;
    volatile uint32_t PLSTS;
    volatile uint32_t VRCTL;
    volatile uint32_t VRSTS;
    volatile uint32_t IOSHCTL;
    volatile uint32_t FSHPCTL;
    volatile uint32_t BATLDCTL;
    volatile const uint32_t RESERVE1[1];
    volatile uint32_t IOTGDBCTL;
    volatile uint32_t GPATGCTL;
    volatile uint32_t GPBTGCTL;
    volatile uint32_t GPCTGCTL;
    volatile uint32_t GPDTGCTL;
    volatile const uint32_t RESERVE2[3];
    volatile uint32_t STMRWKCTL;
    volatile const uint32_t RESERVE3[3];
    volatile uint32_t PINWKCTL;
    volatile const uint32_t RESERVE4[39];
    volatile uint32_t SYSRB0PC;
    volatile uint32_t SYSRB1PC;
    volatile uint32_t SYSRB2PC;
    volatile const uint32_t RESERVE5[1];
    volatile uint32_t LPSYSRPC;

} PMC_T;
# 386 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\psio_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\psio_reg.h"
typedef struct
{
# 145 "../../../../Library/Device/Nuvoton/M55M1/Include\\psio_reg.h"
    volatile uint32_t SCCTL;
    volatile uint32_t SCSLOT;

} SCCT_T;

typedef struct
{
# 499 "../../../../Library/Device/Nuvoton/M55M1/Include\\psio_reg.h"
    volatile uint32_t GENCTL;
    volatile uint32_t DATCTL;
    volatile const uint32_t INSTS;
    volatile const uint32_t INDAT;
    volatile uint32_t OUTDAT;
    volatile uint32_t CPCTL0;
    volatile uint32_t CPCTL1;
    volatile const uint32_t RESERVE0[1];
} GNCT_T;

typedef struct
{
# 995 "../../../../Library/Device/Nuvoton/M55M1/Include\\psio_reg.h"
    volatile uint32_t INTCTL;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile uint32_t TRANSTS;
    volatile uint32_t ISSTS;
    volatile uint32_t PDMACTL;
    volatile uint32_t PODAT;
    volatile uint32_t PIDAT;
    SCCT_T SCCT[4];
    GNCT_T GNCT[8];
} PSIO_T;
# 387 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\qspi_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\qspi_reg.h"
typedef struct
{
# 384 "../../../../Library/Device/Nuvoton/M55M1/Include\\qspi_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t CLKDIV;
    volatile uint32_t SSCTL;
    volatile uint32_t PDMACTL;
    volatile uint32_t FIFOCTL;
    volatile uint32_t STATUS;
    volatile const uint32_t STATUS2;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t TX;
    volatile const uint32_t RESERVE1[3];
    volatile const uint32_t RX;
} QSPI_T;
# 388 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\rtc_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\rtc_reg.h"
typedef struct
{
# 813 "../../../../Library/Device/Nuvoton/M55M1/Include\\rtc_reg.h"
    volatile uint32_t INIT;
    volatile uint32_t RESERVE0;
    volatile uint32_t FREQADJ;
    volatile uint32_t TIME;
    volatile uint32_t CAL;
    volatile uint32_t CLKFMT;
    volatile uint32_t WEEKDAY;
    volatile uint32_t TALM;
    volatile uint32_t CALM;
    volatile const uint32_t LEAPYEAR;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile uint32_t TICK;
    volatile uint32_t TAMSK;
    volatile uint32_t CAMSK;
    volatile uint32_t SPRCTL;
    volatile uint32_t SPR[20];
    volatile const uint32_t RESERVE1[28];
    volatile uint32_t LXTCTL;
    volatile uint32_t GPIOCTL0;
    volatile uint32_t GPIOCTL1;
    volatile const uint32_t RESERVE2[1];
    volatile uint32_t DSTCTL;
    volatile const uint32_t RESERVE3[3];
    volatile uint32_t TAMPCTL;
    volatile const uint32_t RESERVE4[1];
    volatile uint32_t TAMPSEED;
    volatile const uint32_t RESERVE5[1];
    volatile const uint32_t TAMPTIME;
    volatile const uint32_t TAMPCAL;
    volatile const uint32_t RESERVE6[2];
    volatile uint32_t CLKDCTL;
    volatile uint32_t CDBR;
} RTC_T;
# 389 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\sc_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\sc_reg.h"
typedef struct
{
# 685 "../../../../Library/Device/Nuvoton/M55M1/Include\\sc_reg.h"
    volatile uint32_t DAT;
    volatile uint32_t CTL;
    volatile uint32_t ALTCTL;
    volatile uint32_t EGT;
    volatile uint32_t RXTOUT;
    volatile uint32_t ETUCTL;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile uint32_t STATUS;
    volatile uint32_t PINCTL;
    volatile uint32_t TMRCTL0;
    volatile uint32_t TMRCTL1;
    volatile uint32_t TMRCTL2;
    volatile uint32_t UARTCTL;
    volatile const uint32_t RESERVE0[5];
    volatile uint32_t ACTCTL;

} SC_T;
# 390 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\scu_reg.h" 1
# 27 "../../../../Library/Device/Nuvoton/M55M1/Include\\scu_reg.h"
typedef struct
{
# 273 "../../../../Library/Device/Nuvoton/M55M1/Include\\scu_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t STS;
    volatile const uint32_t RESERVE0[2];
    volatile uint32_t SPW[4];
    volatile const uint32_t RESERVE1[12];
    volatile uint32_t NSCTL;
    volatile uint32_t NSSTS;
    volatile const uint32_t RESERVE2[2];
    volatile uint32_t NSPW[4];
    volatile const uint32_t RESERVE3[32];
    volatile uint32_t MISC;
    volatile const uint32_t RESERVE4[2];
    volatile const uint32_t ID;
} DPM_T;
# 568 "../../../../Library/Device/Nuvoton/M55M1/Include\\scu_reg.h"
typedef struct
{
# 602 "../../../../Library/Device/Nuvoton/M55M1/Include\\scu_reg.h"
    volatile uint32_t CTL;
    volatile const uint32_t STS;
} PLM_T;
# 635 "../../../../Library/Device/Nuvoton/M55M1/Include\\scu_reg.h"
typedef struct
{
# 2267 "../../../../Library/Device/Nuvoton/M55M1/Include\\scu_reg.h"
    union
    {
        volatile uint32_t DxPNSy[19];
        struct
        {
            volatile uint32_t D0PNS0;
            volatile const uint32_t RESERVE0[1];
            volatile uint32_t D0PNS2;
            volatile const uint32_t RESERVE1[5];
            volatile uint32_t D1PNS0;
            volatile uint32_t D1PNS1;
            volatile uint32_t D1PNS2;
            volatile const uint32_t RESERVE2[1];
            volatile uint32_t D1PNS4;
            volatile const uint32_t RESERVE3[3];
            volatile uint32_t D2PNS0;
            volatile const uint32_t RESERVE4[1];
            volatile uint32_t D2PNS2;
        };
    };
    volatile const uint32_t RESERVE5[5];
    volatile const uint32_t FNSADDR;
    volatile uint32_t EINTNS;
    volatile const uint32_t RESERVE6[2];
    volatile uint32_t IONS[10];
    volatile const uint32_t RESERVE7[26];
    union
    {
        volatile uint32_t SVIEN[2];
        struct
        {
            volatile uint32_t SVIEN0;
            volatile uint32_t SVIEN1;
        };
    };
    volatile const uint32_t RESERVE8[2];
    union
    {
        volatile uint32_t SVINTSTS[3];
        struct
        {
            volatile uint32_t SVINTSTS0;
            volatile uint32_t SVINTSTS1;
            volatile const uint32_t SVINTSTS2;
        };
    };
    volatile const uint32_t RESERVE9[1];
    union
    {
        volatile const uint32_t PVSRC[17];
        struct
        {
            volatile const uint32_t APB0PVSRC;
            volatile const uint32_t APB1PVSRC;
            volatile const uint32_t APB2PVSRC;
            volatile const uint32_t APB3PVSRC;
            volatile const uint32_t APB4PVSRC;
            volatile const uint32_t APB5PVSRC;
            volatile const uint32_t RESERVE10[2];
            volatile const uint32_t D0PPC0PVSRC;
            volatile const uint32_t D1PPC0PVSRC;
            volatile const uint32_t D1PPC1PVSRC;
            volatile const uint32_t D2PPC0PVSRC;
            volatile const uint32_t RESERVE11[4];
            volatile const uint32_t EBIPVSRC;
        };
    };
    volatile const uint32_t RESERVE12[15];
    union
    {
        volatile const uint32_t PMVA[64];
        struct
        {
            volatile const uint32_t APB0PVA;
            volatile const uint32_t APB1PVA;
            volatile const uint32_t APB2PVA;
            volatile const uint32_t APB3PVA;
            volatile const uint32_t APB4PVA;
            volatile const uint32_t APB5PVA;
            volatile const uint32_t RESERVE13[2];
            volatile const uint32_t D0PPC0PVA;
            volatile const uint32_t D1PPC0PVA;
            volatile const uint32_t D1PPC1PVA;
            volatile const uint32_t D2PPC0PVA;
            volatile const uint32_t RESERVE14[4];
            volatile const uint32_t EBIPVA;
            volatile const uint32_t RESERVE15[15];
            volatile const uint32_t GDMAMVA;
            volatile const uint32_t PDMA0MVA;
            volatile const uint32_t PDMA1MVA;
            volatile const uint32_t USBH0MVA;
            volatile const uint32_t RESERVE16[1];
            volatile const uint32_t HSUSBHMVA;
            volatile const uint32_t HSUSBDMVA;
            volatile const uint32_t SDH0MVA;
            volatile const uint32_t SDH1MVA;
            volatile const uint32_t EMACMVA;
            volatile const uint32_t CRYPTOMVA;
            volatile const uint32_t CRCMVA;
            volatile const uint32_t LPPDMAMVA;
            volatile const uint32_t CCAPMVA;
            volatile const uint32_t NPUIF1MVA;
            volatile const uint32_t NPUIF0MVA;
            volatile const uint32_t SPIM0MVA;
            volatile const uint32_t SPIM1MVA;
            volatile const uint32_t RESERVE17[14];
        };
    };
    union
    {
        volatile uint32_t DxPNPy[19];
        struct
        {
            volatile uint32_t D0PNP0;
            volatile const uint32_t RESERVE18[1];
            volatile uint32_t D0PNP2;
            volatile const uint32_t RESERVE19[5];
            volatile uint32_t D1PNP0;
            volatile uint32_t D1PNP1;
            volatile uint32_t D1PNP2;
            volatile const uint32_t RESERVE20[1];
            volatile uint32_t D1PNP4;
            volatile const uint32_t RESERVE21[3];
            volatile uint32_t D2PNP0;
            volatile const uint32_t RESERVE22[1];
            volatile uint32_t D2PNP2;
        };
    };
    volatile const uint32_t RESERVE23[45];
    volatile uint32_t IONP;
    volatile const uint32_t RESERVE24[7];
    volatile uint32_t SINFAEN;
    volatile uint32_t SCWP;
    volatile const uint32_t RESERVE25[14];
    volatile uint32_t NSMCTL;
    volatile uint32_t NSMLOAD;
    volatile uint32_t NSMVAL;
    volatile uint32_t NSMSTS;
    volatile const uint32_t SRAM0MPCLUT0;
    volatile const uint32_t SRAM1MPCLUT0;
    volatile const uint32_t SRAM2MPCLUT0;
    volatile const uint32_t RESERVE26[1];
    volatile const uint32_t SRAM3MPCLUT0;
    volatile const uint32_t LPSRAMMPCLUT0;
    volatile const uint32_t RESERVE27[2];
    volatile const uint32_t SPIM0MPCLUT[8];
    volatile const uint32_t SPIM1MPCLUT[8];
} SCU_T;
# 391 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\sdh_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\sdh_reg.h"
typedef struct
{
# 342 "../../../../Library/Device/Nuvoton/M55M1/Include\\sdh_reg.h"
    volatile uint32_t FB[32];

    volatile const uint32_t RESERVE0[224];

    volatile uint32_t DMACTL;

    volatile const uint32_t RESERVE1[1];

    volatile uint32_t DMASA;
    volatile const uint32_t DMABCNT;
    volatile uint32_t DMAINTEN;
    volatile uint32_t DMAINTSTS;

    volatile const uint32_t RESERVE2[250];

    volatile uint32_t GCTL;
    volatile uint32_t GINTEN;
    volatile uint32_t GINTSTS;

    volatile const uint32_t RESERVE3[5];

    volatile uint32_t CTL;
    volatile uint32_t CMDARG;
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile const uint32_t RESP0;
    volatile const uint32_t RESP1;
    volatile uint32_t BLEN;
    volatile uint32_t TOUT;

} SDH_T;
# 392 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\spi_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\spi_reg.h"
typedef struct
{
# 534 "../../../../Library/Device/Nuvoton/M55M1/Include\\spi_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t CLKDIV;
    volatile uint32_t SSCTL;
    volatile uint32_t PDMACTL;
    volatile uint32_t FIFOCTL;
    volatile uint32_t STATUS;
    volatile const uint32_t STATUS2;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t TX;
    volatile const uint32_t RESERVE1[3];
    volatile const uint32_t RX;
    volatile const uint32_t RESERVE2[11];
    volatile uint32_t I2SCTL;
    volatile uint32_t I2SCLK;
    volatile uint32_t I2SSTS;
} SPI_T;
# 393 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\spim_reg.h" 1
# 31 "../../../../Library/Device/Nuvoton/M55M1/Include\\spim_reg.h"
typedef struct
{
# 878 "../../../../Library/Device/Nuvoton/M55M1/Include\\spim_reg.h"
    volatile uint32_t CTL0;
    volatile uint32_t CTL1;
    volatile const uint32_t RESERVE0[1];
    volatile uint32_t RXCLKDLY;
    volatile const uint32_t RX[4];




    volatile uint32_t TX[4];




    volatile uint32_t SRAMADDR;
    volatile uint32_t DMACNT;
    volatile uint32_t FADDR;
    volatile const uint32_t RESERVE1[2];
    volatile uint32_t DMMCTL;
    volatile uint32_t CTL2;
    volatile uint32_t CMDCODE;
    volatile uint32_t MODE;
    volatile uint32_t PHDMAW;
    volatile uint32_t PHDMAR;
    volatile uint32_t PHDMM;
    volatile const uint32_t RESERVE2[2];
    volatile uint32_t DLL0;
    volatile uint32_t DLL1;
    volatile uint32_t DLL2;
    volatile const uint32_t RESERVE3[3];
    volatile uint32_t HYPER_CMD;
    volatile uint32_t HYPER_CONFIG1;
    volatile uint32_t HYPER_CONFIG2;
    volatile uint32_t HYPER_ADR;
    volatile uint32_t HYPER_WDATA;
    volatile const uint32_t HYPER_RDATA;
    volatile uint32_t HYPER_INTEN;
    volatile uint32_t HYPER_INTSTS;
} SPIM_T;
# 394 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\sys_reg.h" 1
# 27 "../../../../Library/Device/Nuvoton/M55M1/Include\\sys_reg.h"
typedef struct
{
# 2931 "../../../../Library/Device/Nuvoton/M55M1/Include\\sys_reg.h"
    volatile const uint32_t PDID;
    volatile uint32_t RSTCTL;
    volatile uint32_t RSTSTS;
    volatile uint32_t VTORSET;
    volatile const uint32_t RESERVE0[4];
    volatile uint32_t BODCTL;
    volatile uint32_t PORCTL;
    volatile uint32_t VREFCTL;
    volatile uint32_t IVSCTL;
    volatile uint32_t USBPHY;
    volatile uint32_t UTCPDCTL;
    volatile const uint32_t RESERVE1[18];
    volatile uint32_t GPA_MFOS;
    volatile uint32_t GPB_MFOS;
    volatile uint32_t GPC_MFOS;
    volatile uint32_t GPD_MFOS;
    volatile uint32_t GPE_MFOS;
    volatile uint32_t GPF_MFOS;
    volatile uint32_t GPG_MFOS;
    volatile uint32_t GPH_MFOS;
    volatile uint32_t GPI_MFOS;
    volatile uint32_t GPJ_MFOS;
    volatile const uint32_t RESERVE2[22];
    volatile uint32_t REGLCTL;
    volatile const uint32_t RESERVE3[3];
    volatile uint32_t TCTL48M;
    volatile uint32_t TIEN48M;
    volatile uint32_t TISTS48M;
    volatile const uint32_t RESERVE4[1];
    volatile uint32_t TCTL12M;
    volatile uint32_t TIEN12M;
    volatile uint32_t TISTS12M;
    volatile const uint32_t RESERVE5[1];
    volatile uint32_t TCTLMIRC;
    volatile uint32_t TIENMIRC;
    volatile uint32_t TISTSMIRC;
    volatile const uint32_t RESERVE6[49];
    volatile uint32_t ACMPRST;
    volatile uint32_t AWFRST;
    volatile uint32_t BPWMRST;
    volatile uint32_t CANFDRST;
    volatile uint32_t CCAPRST;
    volatile uint32_t CRCRST;
    volatile uint32_t CRYPTORST;
    volatile uint32_t DACRST;
    volatile uint32_t DMICRST;
    volatile uint32_t EADCRST;
    volatile uint32_t EBIRST;
    volatile uint32_t ECAPRST;
    volatile uint32_t EMACRST;
    volatile uint32_t EPWMRST;
    volatile uint32_t EQEIRST;
    volatile uint32_t FMCRST;
    volatile uint32_t GDMARST;
    volatile uint32_t GPIORST;
    volatile uint32_t HSOTGRST;
    volatile uint32_t HSUSBDRST;
    volatile uint32_t HSUSBHRST;
    volatile uint32_t I2CRST;
    volatile uint32_t I2SRST;
    volatile uint32_t I3CRST;
    volatile uint32_t KDFRST;
    volatile uint32_t KPIRST;
    volatile uint32_t KSRST;
    volatile uint32_t LPADCRST;
    volatile uint32_t LPPDMARST;
    volatile uint32_t LPGPIORST;
    volatile uint32_t LPI2CRST;
    volatile uint32_t LPSPIRST;
    volatile uint32_t LPTMRRST;
    volatile uint32_t LPUARTRST;
    volatile uint32_t OTFCRST;
    volatile uint32_t OTGRST;
    volatile uint32_t PDMARST;
    volatile uint32_t PSIORST;
    volatile uint32_t QSPIRST;
    volatile uint32_t RTCRST;
    volatile uint32_t SCRST;
    volatile uint32_t SCURST;
    volatile uint32_t SDHRST;
    volatile uint32_t SPIRST;
    volatile uint32_t SPIMRST;
    volatile const uint32_t RESERVE7[3];
    volatile uint32_t TMRRST;
    volatile uint32_t TRNGRST;
    volatile uint32_t TTMRRST;
    volatile uint32_t UARTRST;
    volatile uint32_t USBDRST;
    volatile uint32_t USBHRST;
    volatile uint32_t USCIRST;
    volatile uint32_t UTCPDRST;
    volatile const uint32_t RESERVE8[8];
    volatile uint32_t GPA_MFP0;
    volatile uint32_t GPA_MFP1;
    volatile uint32_t GPA_MFP2;
    volatile uint32_t GPA_MFP3;
    volatile uint32_t GPB_MFP0;
    volatile uint32_t GPB_MFP1;
    volatile uint32_t GPB_MFP2;
    volatile uint32_t GPB_MFP3;
    volatile uint32_t GPC_MFP0;
    volatile uint32_t GPC_MFP1;
    volatile uint32_t GPC_MFP2;
    volatile uint32_t GPC_MFP3;
    volatile uint32_t GPD_MFP0;
    volatile uint32_t GPD_MFP1;
    volatile uint32_t GPD_MFP2;
    volatile uint32_t GPD_MFP3;
    volatile uint32_t GPE_MFP0;
    volatile uint32_t GPE_MFP1;
    volatile uint32_t GPE_MFP2;
    volatile uint32_t GPE_MFP3;
    volatile uint32_t GPF_MFP0;
    volatile uint32_t GPF_MFP1;
    volatile uint32_t GPF_MFP2;
    volatile const uint32_t RESERVE9[1];
    volatile uint32_t GPG_MFP0;
    volatile uint32_t GPG_MFP1;
    volatile uint32_t GPG_MFP2;
    volatile uint32_t GPG_MFP3;
    volatile uint32_t GPH_MFP0;
    volatile uint32_t GPH_MFP1;
    volatile uint32_t GPH_MFP2;
    volatile uint32_t GPH_MFP3;
    volatile const uint32_t RESERVE10[1];
    volatile uint32_t GPI_MFP1;
    volatile uint32_t GPI_MFP2;
    volatile uint32_t GPI_MFP3;
    volatile uint32_t GPJ_MFP0;
    volatile uint32_t GPJ_MFP1;
    volatile uint32_t GPJ_MFP2;
    volatile uint32_t GPJ_MFP3;
    volatile const uint32_t RESERVE11[88];
    volatile uint32_t NMIEN;
    volatile const uint32_t NMISTS;

} SYS_T;
# 395 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\timer_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\timer_reg.h"
typedef struct
{
# 809 "../../../../Library/Device/Nuvoton/M55M1/Include\\timer_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t CMP;
    volatile uint32_t INTSTS;
    volatile uint32_t CNT;
    volatile const uint32_t CAP;
    volatile uint32_t EXTCTL;
    volatile uint32_t EINTSTS;
    volatile uint32_t TRGCTL;
    volatile uint32_t ALTCTL;
    volatile uint32_t CAPNF;

    volatile const uint32_t RESERVE0[6];

    volatile uint32_t PWMCTL;
    volatile uint32_t PWMCLKSRC;
    volatile uint32_t PWMCLKPSC;
    volatile uint32_t PWMCNTCLR;
    volatile uint32_t PWMPERIOD;
    volatile uint32_t PWMCMPDAT;
    volatile uint32_t PWMDTCTL;
    volatile const uint32_t PWMCNT;
    volatile uint32_t PWMMSKEN;
    volatile uint32_t PWMMSK;
    volatile uint32_t PWMBNF;
    volatile uint32_t PWMFAILBRK;
    volatile uint32_t PWMBRKCTL;
    volatile uint32_t PWMPOLCTL;
    volatile uint32_t PWMPOEN;
    volatile uint32_t PWMSWBRK;
    volatile uint32_t PWMINTEN0;
    volatile uint32_t PWMINTEN1;
    volatile uint32_t PWMINTSTS0;
    volatile uint32_t PWMINTSTS1;
    volatile uint32_t PWMTRGCTL;
    volatile uint32_t PWMSCTL;
    volatile uint32_t PWMSTRG;
    volatile uint32_t PWMSTATUS;
    volatile const uint32_t PWMPBUF;
    volatile const uint32_t PWMCMPBUF;
    volatile uint32_t PWMIFA;
    volatile uint32_t PWMAINTSTS;
    volatile uint32_t PWMAINTEN;
    volatile uint32_t PWMAPDMACTL;
    volatile uint32_t PWMEXTETCTL;

} TIMER_T;
# 396 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\trng_reg.h" 1
# 26 "../../../../Library/Device/Nuvoton/M55M1/Include\\trng_reg.h"
typedef struct
{
# 136 "../../../../Library/Device/Nuvoton/M55M1/Include\\trng_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t CFG;
    volatile const uint32_t STS;
    volatile const uint32_t DATA_OUT[4];




} TRNG_T;
# 397 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\ttmr_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\ttmr_reg.h"
typedef struct
{
# 251 "../../../../Library/Device/Nuvoton/M55M1/Include\\ttmr_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t CMP;
    volatile uint32_t INTSTS;
    volatile uint32_t CNT;
    volatile const uint32_t RESERVE0[3];
    volatile uint32_t TRGCTL;
} TTMR_T;
# 398 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\uart_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\uart_reg.h"
typedef struct
{
# 881 "../../../../Library/Device/Nuvoton/M55M1/Include\\uart_reg.h"
    volatile uint32_t DAT;
    volatile uint32_t INTEN;
    volatile uint32_t FIFO;
    volatile uint32_t LINE;
    volatile uint32_t MODEM;
    volatile uint32_t MODEMSTS;
    volatile uint32_t FIFOSTS;
    volatile uint32_t INTSTS;
    volatile uint32_t TOUT;
    volatile uint32_t BAUD;
    volatile uint32_t IRDA;
    volatile uint32_t ALTCTL;
    volatile uint32_t FUNCSEL;
    volatile uint32_t LINCTL;
    volatile uint32_t LINSTS;
    volatile uint32_t BRCOMP;
    volatile uint32_t WKCTL;
    volatile uint32_t WKSTS;
    volatile uint32_t DWKCOMP;
    volatile uint32_t RS485DD;
    volatile const uint32_t RESERVE0[1003];
    volatile const uint32_t VERSION;

} UART_T;
# 399 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\utcpd_reg.h" 1
# 30 "../../../../Library/Device/Nuvoton/M55M1/Include\\utcpd_reg.h"
typedef struct
{
# 1126 "../../../../Library/Device/Nuvoton/M55M1/Include\\utcpd_reg.h"
    volatile uint32_t VID;
    volatile uint32_t PID;
    volatile uint32_t DID;
    volatile uint32_t TCREV;
    volatile uint32_t PDREV;
    volatile uint32_t IS;
    volatile uint32_t IE;
    volatile uint32_t PWRSTSIE;
    volatile uint32_t FUTSTSIE;
    volatile uint32_t CTL;
    volatile uint32_t PINPL;
    volatile uint32_t ROLCTL;
    volatile uint32_t FUTCTL;
    volatile uint32_t PWRCTL;
    volatile const uint32_t CCSTS;
    volatile const uint32_t PWRSTS;
    volatile const uint32_t FUTSTS;
    volatile uint32_t CMD;
    volatile uint32_t DVCAP1;
    volatile uint32_t DVCAP2;
    volatile uint32_t MSHEAD;
    volatile uint32_t DTRXEVNT;
    volatile const uint32_t RXBCNT;
    volatile const uint32_t RXFTYPE;
    volatile const uint32_t RXHEAD;
    volatile const uint32_t RESERVE0[3];
    volatile const uint32_t RXDA0;
    volatile const uint32_t RXDA1;
    volatile const uint32_t RXDA2;
    volatile const uint32_t RXDA3;
    volatile const uint32_t RXDA4;
    volatile const uint32_t RXDA5;
    volatile const uint32_t RXDA6;
    volatile const uint32_t RESERVE1[1];
    volatile uint32_t TXCTL;
    volatile uint32_t TXBCNT;
    volatile uint32_t TXHEAD;
    volatile const uint32_t RESERVE2[1];
    volatile uint32_t TXDA0;
    volatile uint32_t TXDA1;
    volatile uint32_t TXDA2;
    volatile uint32_t TXDA3;
    volatile uint32_t TXDA4;
    volatile uint32_t TXDA5;
    volatile uint32_t TXDA6;
    volatile const uint32_t RESERVE3[1];
    volatile uint32_t VBVOL;
    volatile uint32_t SKVBDCTH;
    volatile uint32_t SPDGTH;
    volatile uint32_t VBAMH;
    volatile uint32_t VBAML;
    volatile uint32_t VNDIS;
    volatile uint32_t VNDIE;
    volatile uint32_t MUXSEL;
    volatile uint32_t VCDGCTL;
    volatile uint32_t PHYSLEW;
    volatile uint32_t ADGTM;
    volatile uint32_t VSAFE0V;
    volatile uint32_t VSAFE5V;
    volatile uint32_t RATIO;
    volatile uint32_t INTFRAME;
    volatile uint32_t VBOVTH;
    volatile uint32_t VNDINIT;
    volatile uint32_t BMCTXBP;
    volatile uint32_t BMCTXDU;
    volatile uint32_t VCPSVOL;
    volatile uint32_t VCUV;
    volatile const uint32_t RESERVE4[1];
    volatile uint32_t BMCSLICE;
    volatile uint32_t PHYCTL;
    volatile uint32_t FRSRXCTL;
    volatile const uint32_t VCVOL;
    volatile const uint32_t RESERVE5[118];
    volatile uint32_t CLKINFO;

} UTCPD_T;
# 400 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\ui2c_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\ui2c_reg.h"
typedef struct
{
# 374 "../../../../Library/Device/Nuvoton/M55M1/Include\\ui2c_reg.h"
    volatile uint32_t CTL;

    volatile const uint32_t RESERVE0[1];

    volatile uint32_t BRGEN;

    volatile const uint32_t RESERVE1[8];

    volatile uint32_t LINECTL;
    volatile uint32_t TXDAT;
    volatile const uint32_t RXDAT;

    volatile const uint32_t RESERVE2[3];

    volatile uint32_t DEVADDR0;
    volatile uint32_t DEVADDR1;
    volatile uint32_t ADDRMSK0;
    volatile uint32_t ADDRMSK1;
    volatile uint32_t WKCTL;
    volatile uint32_t WKSTS;
    volatile uint32_t PROTCTL;
    volatile uint32_t PROTIEN;
    volatile uint32_t PROTSTS;

    volatile const uint32_t RESERVE3[8];

    volatile uint32_t ADMAT;
    volatile uint32_t TMCTL;

} UI2C_T;
# 401 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\usbd_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\usbd_reg.h"
typedef struct
{
# 102 "../../../../Library/Device/Nuvoton/M55M1/Include\\usbd_reg.h"
    volatile uint32_t BUFSEG;
    volatile uint32_t MXPLD;
    volatile uint32_t CFG;
    volatile uint32_t CFGP;

} USBD_EP_T;

typedef struct
{
# 639 "../../../../Library/Device/Nuvoton/M55M1/Include\\usbd_reg.h"
    volatile uint32_t INTEN;
    volatile uint32_t INTSTS;
    volatile uint32_t FADDR;
    volatile const uint32_t EPSTS;
    volatile uint32_t ATTR;
    volatile const uint32_t VBUSDET;
    volatile uint32_t STBUFSEG;

    volatile const uint32_t RESERVE0[1];

    volatile const uint32_t EPSTS0;
    volatile const uint32_t EPSTS1;
    volatile const uint32_t EPSTS2;
    volatile const uint32_t EPSTS3;
    volatile uint32_t EPINTSTS;

    volatile const uint32_t RESERVE1[21];

    volatile const uint32_t LPMATTR;
    volatile const uint32_t FN;
    volatile uint32_t SE0;

    volatile const uint32_t RESERVE2[283];

    USBD_EP_T EP[25];

} USBD_T;
# 402 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\usbh_reg.h" 1
# 27 "../../../../Library/Device/Nuvoton/M55M1/Include\\usbh_reg.h"
typedef struct
{
# 542 "../../../../Library/Device/Nuvoton/M55M1/Include\\usbh_reg.h"
    volatile const uint32_t HcRevision;
    volatile uint32_t HcControl;
    volatile uint32_t HcCommandStatus;
    volatile uint32_t HcInterruptStatus;
    volatile uint32_t HcInterruptEnable;
    volatile uint32_t HcInterruptDisable;
    volatile uint32_t HcHCCA;
    volatile uint32_t HcPeriodCurrentED;
    volatile uint32_t HcControlHeadED;
    volatile uint32_t HcControlCurrentED;
    volatile uint32_t HcBulkHeadED;
    volatile uint32_t HcBulkCurrentED;
    volatile uint32_t HcDoneHead;
    volatile uint32_t HcFmInterval;
    volatile const uint32_t HcFmRemaining;
    volatile const uint32_t HcFmNumber;
    volatile uint32_t HcPeriodicStart;
    volatile uint32_t HcLSThreshold;
    volatile uint32_t HcRhDescriptorA;
    volatile uint32_t HcRhDescriptorB;
    volatile uint32_t HcRhStatus;
    volatile uint32_t HcRhPortStatus[6];
    volatile const uint32_t RESERVE0[101];
    volatile uint32_t HcPhyControl;
    volatile uint32_t HcMiscControl;

} USBH_T;
# 403 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\uspi_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\uspi_reg.h"
typedef struct
{
# 425 "../../../../Library/Device/Nuvoton/M55M1/Include\\uspi_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t INTEN;
    volatile uint32_t BRGEN;

    volatile const uint32_t RESERVE0[1];

    volatile uint32_t DATIN0;

    volatile const uint32_t RESERVE1[3];

    volatile uint32_t CTLIN0;

    volatile const uint32_t RESERVE2[1];

    volatile uint32_t CLKIN;
    volatile uint32_t LINECTL;
    volatile uint32_t TXDAT;
    volatile const uint32_t RXDAT;
    volatile uint32_t BUFCTL;
    volatile uint32_t BUFSTS;
    volatile uint32_t PDMACTL;

    volatile const uint32_t RESERVE3[4];

    volatile uint32_t WKCTL;
    volatile uint32_t WKSTS;
    volatile uint32_t PROTCTL;
    volatile uint32_t PROTIEN;
    volatile uint32_t PROTSTS;

} USPI_T;
# 404 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\uuart_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\uuart_reg.h"
typedef struct
{
# 424 "../../../../Library/Device/Nuvoton/M55M1/Include\\uuart_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t INTEN;
    volatile uint32_t BRGEN;

    volatile const uint32_t RESERVE0[1];

    volatile uint32_t DATIN0;

    volatile const uint32_t RESERVE1[3];

    volatile uint32_t CTLIN0;

    volatile const uint32_t RESERVE2[1];

    volatile uint32_t CLKIN;
    volatile uint32_t LINECTL;
    volatile uint32_t TXDAT;
    volatile uint32_t RXDAT;
    volatile uint32_t BUFCTL;
    volatile uint32_t BUFSTS;
    volatile uint32_t PDMACTL;

    volatile const uint32_t RESERVE3[4];

    volatile uint32_t WKCTL;
    volatile uint32_t WKSTS;
    volatile uint32_t PROTCTL;
    volatile uint32_t PROTIEN;
    volatile uint32_t PROTSTS;

} UUART_T;
# 405 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\wdt_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\wdt_reg.h"
typedef struct
{
# 133 "../../../../Library/Device/Nuvoton/M55M1/Include\\wdt_reg.h"
    volatile uint32_t CTL;
    volatile uint32_t ALTCTL;
    volatile uint32_t RSTCNT;
    volatile uint32_t STATUS;

} WDT_T;
# 406 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\wwdt_reg.h" 1
# 28 "../../../../Library/Device/Nuvoton/M55M1/Include\\wwdt_reg.h"
typedef struct
{
# 102 "../../../../Library/Device/Nuvoton/M55M1/Include\\wwdt_reg.h"
    volatile uint32_t RLDCNT;
    volatile uint32_t CTL;
    volatile uint32_t STATUS;
    volatile const uint32_t CNT;

} WWDT_T;
# 407 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1483 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h"
typedef volatile uint8_t vu8;
typedef volatile uint16_t vu16;
typedef volatile uint32_t vu32;
typedef volatile uint64_t vu64;
# 1685 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h"
# 1 "../../../../Library/StdDriver/inc\\acmp.h" 1
# 88 "../../../../Library/StdDriver/inc\\acmp.h"
extern int32_t g_ACMP_i32ErrCode;
# 437 "../../../../Library/StdDriver/inc\\acmp.h"
void ACMP_Open(ACMP_T *acmp, uint32_t u32ChNum, uint32_t u32NegSrc, uint32_t u32HysSel);
void ACMP_Close(ACMP_T *acmp, uint32_t u32ChNum);
void ACMP_Calibration(ACMP_T *acmp);
# 1686 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\awf.h" 1
# 116 "../../../../Library/StdDriver/inc\\awf.h"
static __inline void AWF_EnableInt(uint32_t u32IntMask);
static __inline void AWF_DisableInt(uint32_t u32IntMask);
static __inline void AWF_EnableWakeup(uint32_t u32TWKMask);
static __inline void AWF_DisableWakeup(uint32_t u32TWKMask);
# 128 "../../../../Library/StdDriver/inc\\awf.h"
static __inline void AWF_EnableInt(uint32_t u32IntMask)
{
    ((AWF_T *) ((((uint32_t) 0x40000000UL) + 0x00440000UL) + 0x09000UL))->CTL |= u32IntMask;
}
# 140 "../../../../Library/StdDriver/inc\\awf.h"
static __inline void AWF_DisableInt(uint32_t u32IntMask)
{
    ((AWF_T *) ((((uint32_t) 0x40000000UL) + 0x00440000UL) + 0x09000UL))->CTL &= ~u32IntMask;
}
# 152 "../../../../Library/StdDriver/inc\\awf.h"
static __inline void AWF_EnableWakeup(uint32_t u32TWKMask)
{
    ((AWF_T *) ((((uint32_t) 0x40000000UL) + 0x00440000UL) + 0x09000UL))->CTL |= u32TWKMask;
}
# 164 "../../../../Library/StdDriver/inc\\awf.h"
static __inline void AWF_DisableWakeup(uint32_t u32TWKMask)
{
    ((AWF_T *) ((((uint32_t) 0x40000000UL) + 0x00440000UL) + 0x09000UL))->CTL &= ~u32TWKMask;
}

void AWF_Open(uint32_t u32IntEn, uint32_t u32WakeupEn, uint32_t u32HTH, uint32_t u32LTH, uint32_t u32WBINIT, uint32_t u32ACCCount);
void AWF_Close(void);
# 1687 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\bpwm.h" 1
# 329 "../../../../Library/StdDriver/inc\\bpwm.h"
uint32_t BPWM_ConfigCaptureChannel(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32UnitTimeNsec, uint32_t u32CaptureEdge);
uint32_t BPWM_ConfigOutputChannel(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Frequency, uint32_t u32DutyCycle);
void BPWM_Start(BPWM_T *bpwm, uint32_t u32ChannelMask);
void BPWM_Stop(BPWM_T *bpwm, uint32_t u32ChannelMask);
void BPWM_ForceStop(BPWM_T *bpwm, uint32_t u32ChannelMask);
void BPWM_EnableADCTrigger(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Condition);
void BPWM_DisableADCTrigger(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_ClearADCTriggerFlag(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Condition);
uint32_t BPWM_GetADCTriggerFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_EnableCapture(BPWM_T *bpwm, uint32_t u32ChannelMask);
void BPWM_DisableCapture(BPWM_T *bpwm, uint32_t u32ChannelMask);
void BPWM_EnableOutput(BPWM_T *bpwm, uint32_t u32ChannelMask);
void BPWM_DisableOutput(BPWM_T *bpwm, uint32_t u32ChannelMask);
void BPWM_EnableCaptureInt(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Edge);
void BPWM_DisableCaptureInt(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Edge);
void BPWM_ClearCaptureIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Edge);
uint32_t BPWM_GetCaptureIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_EnableDutyInt(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32IntDutyType);
void BPWM_DisableDutyInt(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_ClearDutyIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
uint32_t BPWM_GetDutyIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_EnablePeriodInt(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32IntPeriodType);
void BPWM_DisablePeriodInt(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_ClearPeriodIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
uint32_t BPWM_GetPeriodIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_EnableZeroInt(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_DisableZeroInt(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_ClearZeroIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
uint32_t BPWM_GetZeroIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_EnableLoadMode(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32LoadMode);
void BPWM_DisableLoadMode(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32LoadMode);
void BPWM_SetClockSource(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32ClkSrcSel);
uint32_t BPWM_GetWrapAroundFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
void BPWM_ClearWrapAroundFlag(BPWM_T *bpwm, uint32_t u32ChannelNum);
# 1688 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\canfd.h" 1
# 103 "../../../../Library/StdDriver/inc\\canfd.h"
typedef enum
{
    eCANFD_BYTE8 = 0,
    eCANFD_BYTE12 = 1,
    eCANFD_BYTE16 = 2,
    eCANFD_BYTE20 = 3,
    eCANFD_BYTE24 = 4,
    eCANFD_BYTE32 = 5,
    eCANFD_BYTE48 = 6,
    eCANFD_BYTE64 = 7
} E_CANFD_DATA_FIELD_SIZE;


typedef enum
{
    eCANFD_QUEUE_MODE = 0,
    eCANFD_FIFO_MODE = 1
} E_CANFD_TX_MODE;


typedef struct
{
    E_CANFD_DATA_FIELD_SIZE eDataFieldSize;
    E_CANFD_TX_MODE eModeSel;
    uint32_t u32ElemCnt;
    uint32_t u32DBufNumber;
} CANFD_TX_BUF_CONFIG_T;



typedef struct
{
    uint32_t u32BitRate;
    uint16_t u16TDCOffset;
    uint16_t u16TDCFltrWin;
    uint8_t u8TDC;
} CANFD_NBT_CONFIG_T;



typedef struct
{
    uint32_t u32BitRate;
    uint16_t u16TDCOffset;
    uint16_t u16TDCFltrWin;
    uint8_t u8TDC;
} CANFD_DBT_CONFIG_T;


typedef struct
{
    uint8_t u8PreDivider;
    uint16_t u16NominalPrescaler;
    uint8_t u8NominalRJumpwidth;
    uint8_t u8NominalPhaseSeg1;
    uint8_t u8NominalPhaseSeg2;
    uint8_t u8NominalPropSeg;
    uint8_t u8DataPrescaler;
    uint8_t u8DataRJumpwidth;
    uint8_t u8DataPhaseSeg1;
    uint8_t u8DataPhaseSeg2;
    uint8_t u8DataPropSeg;

} CANFD_TIMEING_CONFIG_T;


typedef struct
{
    CANFD_NBT_CONFIG_T sNormBitRate;
    CANFD_DBT_CONFIG_T sDataBitRate;
    CANFD_TIMEING_CONFIG_T sConfigBitTing;
    uint8_t bFDEn;
    uint8_t bBitRateSwitch;
    uint8_t bEnableLoopBack;
} CANFD_FD_BT_CONFIG_T;


typedef struct
{
    uint32_t u32SIDFC_FLSSA;
    uint32_t u32XIDFC_FLESA;
    uint32_t u32RXF0C_F0SA;
    uint32_t u32RXF1C_F1SA;
    uint32_t u32RXBC_RBSA;
    uint32_t u32TXEFC_EFSA;
    uint32_t u32TXBC_TBSA;
} CANFD_RAM_PART_T;


typedef struct
{
    uint32_t u32SIDFC;
    uint32_t u32XIDFC;
    uint32_t u32RxFifo0;
    uint32_t u32RxFifo1;
    uint32_t u32RxBuf;
    uint32_t u32TxBuf;
    uint32_t u32TxFifoQueue;
    uint32_t u32TxEventFifo;
    uint32_t u32UserDef;
} CANFD_ELEM_SIZE_T;


typedef struct
{
    CANFD_FD_BT_CONFIG_T sBtConfig;
    CANFD_RAM_PART_T sMRamStartAddr;
    CANFD_ELEM_SIZE_T sElemSize;
    CANFD_TX_BUF_CONFIG_T sTxConfig;
    uint32_t u32MRamSize;
} CANFD_FD_T;


typedef enum
{
    eCANFD_SID = 0,
    eCANFD_XID = 1
} E_CANFD_ID_TYPE;


typedef enum
{
    eCANFD_RX_FIFO_0 = 0,
    eCANFD_RX_FIFO_1 = 1,
    eCANFD_RX_DBUF = 2
} E_CANFD_RX_BUF_TYPE;


typedef enum
{
    eCANFD_SYNC = 0,
    eCANFD_IDLE = 1,
    eCANFD_RECEIVER = 2,
    eCANFD_TRANSMITTER = 3
} E_CANFD_COMMUNICATION_STATE;



typedef enum
{
    eCANFD_TRANSMIT_FAIL = 0,
    eCANFD_TRANSMIT_SUCCESS = 1,
    eCANFD_TRANSMIT_TIMEOUT = 2
} E_CANFD_TRANSMIT_STATE;


typedef enum
{
    eCANFD_RECEIVE_EMPTY = 0,
    eCANFD_RECEIVE_SUCCESS = 1,
    eCANFD_RECEIVE_SUCCESS_AND_BUFFER_FULL = 2

} E_CANFD_RECEIVE_STATE;



typedef struct
{
    E_CANFD_RX_BUF_TYPE eRxBuf;
    uint32_t u32BufIdx;
} CANFD_RX_INFO_T;


typedef enum
{
    eCANFD_DATA_FRM = 0,
    eCANFD_REMOTE_FRM = 1
} E_CANFD_FRM_TYPE;


typedef struct
{
    E_CANFD_ID_TYPE eIdType;
    CANFD_RX_INFO_T sRxInfo;
    E_CANFD_FRM_TYPE eFrmType;
    uint32_t u32Id;
    uint32_t u32DLC;
    union
    {
        uint32_t au32Data[(64 / 4)];
        uint8_t au8Data[64];
    };
    uint8_t u8MsgMarker;
    uint16_t u16RxTimestamp;
    uint8_t u8RxTimestampPointer;
    uint8_t bTimestampCaptured;
    uint8_t u8FilterIndex;
    uint8_t bNonMatchingFrame;
    uint8_t bFDFormat;
    uint8_t bBitRateSwitch;
    uint8_t bErrStaInd;
    uint8_t bEvntFifoCon;
} CANFD_FD_MSG_T;



typedef struct
{
    uint32_t u32Id;
    uint32_t u32Config;
    union
    {
        uint32_t au32Data[(64 / 4)];
        uint8_t au8Data[64];
    };
} CANFD_BUF_T;


typedef struct
{
    union
    {
        struct
        {
            uint32_t SFID2 : 11;
            uint32_t reserved1 : 5;
            uint32_t SFID1 : 11;
            uint32_t SFEC : 3;
            uint32_t SFT : 2;
        };
        struct
        {
            uint32_t VALUE;
        };
    };
} CANFD_STD_FILTER_T;


typedef struct
{
    union
    {
        struct
        {
            uint32_t EFID1 : 29;
            uint32_t EFEC : 3;
            uint32_t EFID2 : 29;
            uint32_t reserved1 : 1;
            uint32_t EFT : 2;
        };
        struct
        {
            uint32_t LOWVALUE;
            uint32_t HIGHVALUE;
        };
    };
} CANFD_EXT_FILTER_T;


typedef enum
{
    eCANFD_ACC_NON_MATCH_FRM_RX_FIFO0 = 0x0,
    eCANFD_ACC_NON_MATCH_FRM_RX_FIFO1 = 0x1,
    eCANFD_REJ_NON_MATCH_FRM = 0x3
} E_CANFD_ACC_NON_MATCH_FRM;



typedef enum
{
    eCANFD_SID_FLTR_TYPE_RANGE = 0x0,
    eCANFD_SID_FLTR_TYPE_DUAL = 0x1,
    eCANFD_SID_FLTR_TYPE_CLASSIC = 0x2,
    eCANFD_SID_FLTR_TYPE_DIS = 0x3
} E_CANFD_SID_FLTR_ELEM_TYPE;


typedef enum
{
    eCANFD_XID_FLTR_TYPE_RANGE = 0x0,
    eCANFD_XID_FLTR_TYPE_DUAL = 0x1,
    eCANFD_XID_FLTR_TYPE_CLASSIC = 0x2,
    eCANFD_XID_FLTR_TYPE_RANGE_XIDAM_NOT_APP = 0x3
} E_CANFD_XID_FLTR_ELEM_TYPE;


typedef enum
{
    eCANFD_FLTR_ELEM_DIS = 0x0,
    eCANFD_FLTR_ELEM_STO_FIFO0 = 0x1,
    eCANFD_FLTR_ELEM_STO_FIFO1 = 0x2,
    eCANFD_FLTR_ELEM_REJ_ID = 0x3,
    eCANFD_FLTR_ELEM_SET_PRI = 0x4,
    eCANFD_FLTR_ELEM_SET_PRI_STO_FIFO0 = 0x5,
    eCANFD_FLTR_ELEM_SET_PRI_STO_FIFO1 = 0x6,
    eCANFD_FLTR_ELEM_STO_RX_BUF_OR_DBG_MSG = 0x7
} E_CANFD_FLTR_CONFIG;


typedef struct
{
    E_CANFD_ID_TYPE eIdType;
    uint32_t u32Id;
    uint32_t u32DLC;
    uint32_t u32TxTs;
    uint32_t u32MsgMarker;
    uint8_t bErrStaInd;
    uint8_t bRemote;
    uint8_t bFDFormat;
    uint8_t bBitRateSwitch;
} CANFD_TX_EVNT_ELEM_T;


void CANFD_Open(CANFD_T *canfd, CANFD_FD_T *psCanfdStr);
void CANFD_Close(CANFD_T *canfd);
void CANFD_EnableInt(CANFD_T *canfd, uint32_t u32IntLine0, uint32_t u32IntLine1, uint32_t u32TXBTIE, uint32_t u32TXBCIE);
void CANFD_DisableInt(CANFD_T *canfd, uint32_t u32IntLine0, uint32_t u32IntLine1, uint32_t u32TXBTIE, uint32_t u32TXBCIE);
uint32_t CANFD_TransmitTxMsg(CANFD_T *canfd, uint32_t u32TxBufIdx, CANFD_FD_MSG_T *psTxMsg);
uint32_t CANFD_TransmitDMsg(CANFD_T *canfd, uint32_t u32TxBufIdx, CANFD_FD_MSG_T *psTxMsg);
void CANFD_CopyDataToTransmitBuffer(CANFD_T *psCanfd, uint32_t u32TxBufIdx, CANFD_FD_MSG_T *psTxMsg);
void CANFD_SetGFC(CANFD_T *canfd, E_CANFD_ACC_NON_MATCH_FRM eNMStdFrm, E_CANFD_ACC_NON_MATCH_FRM eEMExtFrm, uint32_t u32RejRmtStdFrm, uint32_t u32RejRmtExtFrm);
void CANFD_SetSIDFltr(CANFD_T *canfd, uint32_t u32FltrIdx, uint32_t u32Filter);
void CANFD_SetXIDFltr(CANFD_T *canfd, uint32_t u32FltrIdx, uint32_t u32FilterLow, uint32_t u32FilterHigh);
uint32_t CANFD_ReadRxBufMsg(CANFD_T *canfd, uint8_t u8MbIdx, CANFD_FD_MSG_T *psMsgBuf);
uint32_t CANFD_ReadRxFifoMsg(CANFD_T *canfd, uint8_t u8FifoIdx, CANFD_FD_MSG_T *psMsgBuf);
void CANFD_CopyDBufToMsgBuf(CANFD_T *canfd, CANFD_BUF_T *psRxBuffer, CANFD_FD_MSG_T *psMsgBuf);
void CANFD_CopyRxFifoToMsgBuf(CANFD_T *canfd, CANFD_BUF_T *psRxBuf, CANFD_FD_MSG_T *psMsgBuf);
uint32_t CANFD_GetRxFifoWaterLvl(CANFD_T *canfd, uint32_t u32RxFifoNum);
uint32_t CANFD_ReadTxFifoEventMsg(CANFD_T *psCanfd, uint8_t u8MbIdx, CANFD_TX_EVNT_ELEM_T *psEventFifoMsgBuf);
void CANFD_TxBufCancelReq(CANFD_T *canfd, uint32_t u32TxBufIdx);
uint32_t CANFD_IsTxBufCancelFin(CANFD_T *canfd, uint32_t u32TxBufIdx);
uint32_t CANFD_IsTxBufTransmitOccur(CANFD_T *canfd, uint32_t u32TxBufIdx);
uint32_t CANFD_GetTxEvntFifoWaterLvl(CANFD_T *canfd);
uint32_t CANFD_WeiteTransmitMsgData(CANFD_T *psCanfd, uint32_t u32TxBufIdx, CANFD_FD_MSG_T *psTxMsg);
void CANFD_InitRxFifo(CANFD_T *canfd, uint32_t u32RxFifoNum, CANFD_RAM_PART_T *psRamConfig, CANFD_ELEM_SIZE_T *psElemSize, uint32_t u32FifoWM, E_CANFD_DATA_FIELD_SIZE eFifoSize);
void CANFD_InitRxDBuf(CANFD_T *canfd, CANFD_RAM_PART_T *psRamConfig, CANFD_ELEM_SIZE_T *psElemSize, E_CANFD_DATA_FIELD_SIZE eRxBufSize);
void CANFD_InitTxDBuf(CANFD_T *canfd, CANFD_RAM_PART_T *psRamConfig, CANFD_ELEM_SIZE_T *psElemSize, E_CANFD_DATA_FIELD_SIZE eTxBufSize);
void CANFD_InitTxEvntFifo(CANFD_T *canfd, CANFD_RAM_PART_T *psRamConfig, CANFD_ELEM_SIZE_T *psElemSize, uint32_t u32FifoWaterLvl);
void CANFD_ConfigSIDFC(CANFD_T *canfd, CANFD_RAM_PART_T *psRamConfig, CANFD_ELEM_SIZE_T *psElemSize);
void CANFD_ConfigXIDFC(CANFD_T *canfd, CANFD_RAM_PART_T *psRamConfig, CANFD_ELEM_SIZE_T *psElemSize);
void CANFD_CopyTxEvntFifoToUsrBuf(CANFD_T *canfd, uint32_t u32TxEvntNum, CANFD_TX_EVNT_ELEM_T *psTxEvntElem);
void CANFD_GetBusErrCount(CANFD_T *canfd, uint8_t *pu8TxErrBuf, uint8_t *pu8RxErrBuf);
void CANFD_RunToNormal(CANFD_T *canfd, uint8_t u8Enable);
void CANFD_GetDefaultConfig(CANFD_FD_T *psConfig, uint8_t u8OpMode);
void CANFD_ClearStatusFlag(CANFD_T *canfd, uint32_t u32InterruptFlag);
uint32_t CANFD_GetStatusFlag(CANFD_T *canfd, uint32_t u32IntTypeFlag);
# 1689 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\ccap.h" 1
# 332 "../../../../Library/StdDriver/inc\\ccap.h"
static __inline void CCAP_SetPacketBuf(uint32_t u32Address)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->PKTBA0 = u32Address;
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->CTL |= (0x1ul << (20));
}




static __inline void CCAP_Close(void)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->CTL &= ~(1ul << (0));
}
# 360 "../../../../Library/StdDriver/inc\\ccap.h"
static __inline void CCAP_EnableInt(uint32_t u32IntMask)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->INTEN = (((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->INTEN & ~((0x1ul << (0)) | (0x1ul << (1)) | (0x1ul << (3)) | (0x3ul << (4))))
                  | u32IntMask;
}
# 380 "../../../../Library/StdDriver/inc\\ccap.h"
static __inline void CCAP_DisableInt(uint32_t u32IntMask)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->INTEN = (((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->INTEN & ~(u32IntMask));
}
# 397 "../../../../Library/StdDriver/inc\\ccap.h"
static __inline uint32_t CCAP_IsIntEnabled(uint32_t u32IntMask)
{
    return (((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->INTEN & u32IntMask);
}
# 412 "../../../../Library/StdDriver/inc\\ccap.h"
static __inline void CCAP_EnableMono(uint32_t u32Interface)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->CTL = (((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->CTL & ~((0x1ul << (18)) | (0x1ul << (17)))) | (0x1ul << (7)) | u32Interface;
}




static __inline void CCAP_DisableMono(void)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->CTL &= ~(0x1ul << (7));
}




static __inline void CCAP_EnableUpdate(void)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->CTL |= (1ul << (20));
}
# 440 "../../../../Library/StdDriver/inc\\ccap.h"
static __inline void CCAP_EnableLumaYOne(uint32_t u32Threshold)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->CTL |= (0x1ul << (19));
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->LUMA_Y1_THD = u32Threshold & 0xff;
}




static __inline void CCAP_DisableLumaYOne(void)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->CTL &= ~(0x1ul << (19));
}




static __inline void CCAP_Start(void)
{
    ((CCAP_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x01000UL))->CTL |= (1ul << (0));
}

void CCAP_Open(uint32_t u32SensorProp, uint32_t u32InFormat, uint32_t u32OutFormat);
void CCAP_Close(void);
void CCAP_SetCroppingWindow(uint32_t u32VStart, uint32_t u32HStart, uint32_t u32Height, uint32_t u32Width);
void CCAP_SetPacketBuf(uint32_t u32Address);
void CCAP_SetPacketScaling(uint32_t u32VNumerator, uint32_t u32VDenominator, uint32_t u32HNumerator, uint32_t u32HDenominator);
void CCAP_SetPacketStride(uint32_t u32Stride);
void CCAP_EnableInt(uint32_t u32IntMask);
void CCAP_DisableInt(uint32_t u32IntMask);
void CCAP_Start(void);
int32_t CCAP_Stop(uint32_t u32FrameComplete);
void CCAP_EnableMono(uint32_t u32Interface);
void CCAP_DisableMono(void);
void CCAP_EnableLumaYOne(uint32_t u32Threshold);
void CCAP_DisableLumaYOne(void);
int32_t CCAP_MD_SetRegionSensitivity(uint32_t u32Y, uint32_t u32X, uint32_t u32Height, uint32_t u32Width, uint32_t u32Sensitivity);
int32_t CCAP_MD_SetGlobalSensitivity(uint32_t u32Sensitivity);
# 1690 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\clk.h" 1
# 1050 "../../../../Library/StdDriver/inc\\clk.h"
void CLK_DisableCKO(void);
void CLK_EnableCKO(uint32_t u32ClkSrc, uint32_t u32ClkDiv, uint32_t u32ClkDivBy1En);
void CLK_DisableMIRC(void);
uint32_t CLK_EnableMIRC(uint32_t u32MircFreq);
                       uint32_t CLK_GetHXTFreq(void);
                       uint32_t CLK_GetLXTFreq(void);
                       uint32_t CLK_GetMIRCFreq(void);
                       uint32_t CLK_GetSCLKFreq(void);
                       uint32_t CLK_GetACLKFreq(void);
                       uint32_t CLK_GetHCLK0Freq(void);
                       uint32_t CLK_GetHCLK1Freq(void);
                       uint32_t CLK_GetHCLK2Freq(void);
                       uint32_t CLK_GetPCLK0Freq(void);
                       uint32_t CLK_GetPCLK1Freq(void);
                       uint32_t CLK_GetPCLK2Freq(void);
                       uint32_t CLK_GetPCLK3Freq(void);
                       uint32_t CLK_GetPCLK4Freq(void);
                       uint32_t CLK_GetPCLK5Freq(void);
uint32_t CLK_SetCoreClock(uint32_t u32Aclk);
void CLK_SetSCLK(uint32_t u32ClkSrc);
uint32_t CLK_SetBusClock(uint32_t u32SCLKSrc, uint32_t u32PllClkSrc, uint32_t u32PllFreq);
void CLK_SetModuleClock(uint64_t u64ModuleIdx, uint32_t u32ClkSrc, uint32_t u32ClkDiv);
void CLK_SetSysTickClockSrc(uint32_t u32ClkSrc);
void CLK_EnableXtalRC(uint32_t u32ClkMask);
void CLK_DisableXtalRC(uint32_t u32ClkMask);
uint32_t CLK_EnableModuleClock(uint64_t u64ModuleIdx);
void CLK_DisableModuleClock(uint64_t u64ModuleIdx);

uint32_t CLK_WaitModuleClockReady(uint64_t u64ModuleIdx);

uint32_t CLK_EnableAPLL(uint32_t u32APLLClkSrc, uint32_t u32APLLFreq, uint32_t u32APLLSelect);
void CLK_DisableAPLL(uint32_t u32APLLSelect);
uint32_t CLK_WaitClockReady(uint32_t u32ClkMask);
uint32_t CLK_WaitClockDisable(uint32_t u32ClkMask);
void CLK_EnableSysTick(uint32_t u32ClkSrc, uint32_t u32Count);
void CLK_DisableSysTick(void);
                       uint32_t CLK_GetAPLL0ClockFreq(void);
                       uint32_t CLK_GetAPLL1ClockFreq(void);
                       uint32_t CLK_GetModuleClockSource(uint64_t u64ModuleIdx);
                       uint32_t CLK_GetModuleClockDivider(uint64_t u64ModuleIdx);
                       uint32_t CLK_SystemClockUpdate(void);
void CLK_SysTickDelay(uint32_t us);
void CLK_SysTickLongDelay(uint32_t us);
# 1691 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\crc.h" 1
# 110 "../../../../Library/StdDriver/inc\\crc.h"
void CRC_Open(uint32_t u32Mode, uint32_t u32Attribute, uint32_t u32Seed, uint32_t u32DataLen);
uint32_t CRC_GetChecksum(void);
# 1692 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\dac.h" 1
# 265 "../../../../Library/StdDriver/inc\\dac.h"
void DAC_Open(DAC_T *dac, uint32_t u32Ch, uint32_t u32TrgSrc);
void DAC_Close(DAC_T *dac, uint32_t u32Ch);
uint32_t DAC_SetDelayTime(DAC_T *dac, uint32_t u32Delay);
# 1693 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\dmic.h" 1
# 95 "../../../../Library/StdDriver/inc\\dmic.h"
typedef struct
{
    uint16_t u16BIQCoeffA1;
    uint16_t u16BIQCoeffA2;
    uint16_t u16BIQCoeffB0;
    uint16_t u16BIQCoeffB1;
    uint16_t u16BIQCoeffB2;
} DMIC_VAD_BIQ_T;
# 369 "../../../../Library/StdDriver/inc\\dmic.h"
static __inline void DMIC_SetFIFOWidth(DMIC_T *dmic, uint32_t u32Width);
static __inline void DMIC_SetGainStep(DMIC_T *dmic, uint32_t u32Volume);
static __inline void DMIC_ResetDSP(DMIC_T *dmic);
static __inline void DMIC_EnableHPF(DMIC_T *dmic, uint32_t u32ChHPF);
static __inline void DMIC_DisableHPF(DMIC_T *dmic, uint32_t u32ChHPF);
static __inline void DMIC_EnableMute(DMIC_T *dmic, uint32_t u32ChMute);
static __inline void DMIC_DisableMute(DMIC_T *dmic, uint32_t u32ChMute);
static __inline uint32_t DMIC_GetFIFOPTR(DMIC_T *dmic);
static __inline void DMIC_VAD_SetBIQCoeff(VAD_T *vad, DMIC_VAD_BIQ_T *psBIQCoeff);
# 388 "../../../../Library/StdDriver/inc\\dmic.h"
static __inline void DMIC_SetFIFOWidth(DMIC_T *dmic, uint32_t u32Width)
{
    dmic->CTL = (dmic->CTL & ~((0x1ul << (31)))) | (u32Width);
}
# 404 "../../../../Library/StdDriver/inc\\dmic.h"
static __inline void DMIC_SetGainStep(DMIC_T *dmic, uint32_t u32Volume)
{
    dmic->CTL = (dmic->CTL & ~((0x3ul << (27)))) | (u32Volume);
}







static __inline void DMIC_ResetDSP(DMIC_T *dmic)
{
    uint32_t u32Delay;
    dmic->CTL |= (0x1ul << (24));
    u32Delay = SystemCoreClock >> 3;

    while ((dmic->CTL & (0x1ul << (24))) && (--u32Delay))
    {
        __builtin_arm_nop();
    }
}
# 436 "../../../../Library/StdDriver/inc\\dmic.h"
static __inline void DMIC_EnableHPF(DMIC_T *dmic, uint32_t u32ChHPF)
{
    dmic->CTL |= (u32ChHPF);
}
# 450 "../../../../Library/StdDriver/inc\\dmic.h"
static __inline void DMIC_DisableHPF(DMIC_T *dmic, uint32_t u32ChHPF)
{
    dmic->CTL &= ~(u32ChHPF);
}
# 466 "../../../../Library/StdDriver/inc\\dmic.h"
static __inline void DMIC_EnableMute(DMIC_T *dmic, uint32_t u32ChMute)
{
    dmic->CTL |= (u32ChMute);
}
# 482 "../../../../Library/StdDriver/inc\\dmic.h"
static __inline void DMIC_DisableMute(DMIC_T *dmic, uint32_t u32ChMute)
{
    dmic->CTL &= ~(u32ChMute);
}
# 496 "../../../../Library/StdDriver/inc\\dmic.h"
static __inline uint32_t DMIC_GetFIFOPTR(DMIC_T *dmic)
{
    return ((dmic->STATUS & (0x1ful << (4))) >> (4));
}







static __inline void DMIC_VAD_SetBIQCoeff(VAD_T *vad, DMIC_VAD_BIQ_T *psBIQCoeff)
{
    vad->BIQCTL0 = (vad->BIQCTL0 & ~(0xfffful << (0))) | ((psBIQCoeff->u16BIQCoeffA1 << (0))&(0xfffful << (0)));
    vad->BIQCTL0 = (vad->BIQCTL0 & ~(0xfffful << (16))) | ((psBIQCoeff->u16BIQCoeffA2 << (16))&(0xfffful << (16)));
    vad->BIQCTL1 = (vad->BIQCTL1 & ~(0xfffful << (0))) | ((psBIQCoeff->u16BIQCoeffB0 << (0))&(0xfffful << (0)));
    vad->BIQCTL1 = (vad->BIQCTL1 & ~(0xfffful << (16))) | ((psBIQCoeff->u16BIQCoeffB1 << (16))&(0xfffful << (16)));
    vad->BIQCTL2 = (vad->BIQCTL2 & ~(0xfffful << (0))) | ((psBIQCoeff->u16BIQCoeffB2 << (0))&(0xfffful << (0)));
}

void DMIC_Open(DMIC_T *dmic);
void DMIC_Close(DMIC_T *dmic);
void DMIC_SetDSPGainVolume(DMIC_T *dmic, uint32_t u32ChMsk, int16_t i16ChVolume);
void DMIC_ClearFIFO(DMIC_T *dmic);
uint32_t DMIC_SetSampleRate(DMIC_T *dmic, uint32_t u32SampleRate);
uint32_t DMIC_GetSampleRate(DMIC_T *dmic);

uint32_t DMIC_VAD_SetSampleRate(VAD_T *vad, uint32_t u32SampleRate);
uint32_t DMIC_VAD_GetSampleRate(VAD_T *vad);
# 1694 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\eadc.h" 1
# 304 "../../../../Library/StdDriver/inc\\eadc.h"
static __inline uint32_t EADC_GET_DATA_VALID_FLAG(EADC_T *eadc, uint32_t u32ModuleMask)
{
    uint32_t u32VALID = 0;

    u32VALID = (eadc->STATUS0 & (0xffffUL << (0)));
    u32VALID |= ((eadc->STATUS1 & (0xfffUL << (0))) << 16);

    return (u32VALID & u32ModuleMask);
}
# 324 "../../../../Library/StdDriver/inc\\eadc.h"
static __inline uint32_t EADC_GET_DATA_OVERRUN_FLAG(EADC_T *eadc, uint32_t u32ModuleMask)
{
    uint32_t u32OVFlag = 0;

    u32OVFlag = ((eadc->STATUS0 & (0xffffUL << (16))) >> (16));
    u32OVFlag |= (eadc->STATUS1 & (0xfffUL << (16)));

    return (u32OVFlag & u32ModuleMask);
}
# 706 "../../../../Library/StdDriver/inc\\eadc.h"
void EADC_Open(EADC_T *eadc, uint32_t u32InputMode);
void EADC_Close(EADC_T *eadc);
void EADC_ConfigSampleModule(EADC_T *eadc, uint32_t u32ModuleNum, uint32_t u32TriggerSrc, uint32_t u32Channel);
void EADC_SetTriggerDelayTime(EADC_T *eadc, uint32_t u32ModuleNum, uint32_t u32TriggerDelayTime, uint32_t u32DelayClockDivider);
void EADC_SetExtendSampleTime(EADC_T *eadc, uint32_t u32ModuleNum, uint32_t u32ExtendSampleTime);
# 1695 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\ebi.h" 1
# 337 "../../../../Library/StdDriver/inc\\ebi.h"
void EBI_Open(uint32_t u32Bank, uint32_t u32DataWidth, uint32_t u32TimingClass, uint32_t u32BusMode, uint32_t u32CSActiveLevel);
void EBI_Close(uint32_t u32Bank);
void EBI_SetBusTiming(uint32_t u32Bank, uint32_t u32TimingConfig, uint32_t u32MclkDiv);
# 1696 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\ecap.h" 1
# 504 "../../../../Library/StdDriver/inc\\ecap.h"
void ECAP_Open(ECAP_T *ecap, uint32_t u32FuncMask);
void ECAP_Close(ECAP_T *ecap);
void ECAP_EnableINT(ECAP_T *ecap, uint32_t u32Mask);
void ECAP_DisableINT(ECAP_T *ecap, uint32_t u32Mask);
# 1697 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\epwm.h" 1
# 298 "../../../../Library/StdDriver/inc\\epwm.h"
static __inline void EPWM_DISABLE_TIMER_SYNC(EPWM_T *epwm, uint32_t u32ChannelMask)
{
    do
    {
        int i;

        for (i = 0; i < 6; i++)
        {
            if ((u32ChannelMask) & (1 << i))
                (epwm)->SSCTL &= ~(1UL << i);
        }
    } while (0);
}
# 465 "../../../../Library/StdDriver/inc\\epwm.h"
static __inline void EPWM_SET_ALIGNED_TYPE(EPWM_T *epwm, uint32_t u32ChannelMask, uint32_t u32AlignedType)
{
    do
    {
        int i;

        for (i = 0; i < 6; i++)
        {
            if ((u32ChannelMask) & (1 << i))
                (epwm)->CTL1 = (((epwm)->CTL1 & ~(3UL << (i << 1))) | ((u32AlignedType) << (i << 1)));
        }
    } while (0);
}
# 541 "../../../../Library/StdDriver/inc\\epwm.h"
static __inline void EPWM_SET_OUTPUT_LEVEL(EPWM_T *epwm, uint32_t u32ChannelMask, uint32_t u32ZeroLevel, uint32_t u32CmpUpLevel, uint32_t u32PeriodLevel, uint32_t u32CmpDownLevel)
{
    do
    {
        int i;

        for (i = 0; i < 6; i++)
        {
            if ((u32ChannelMask) & (1 << i))
            {
                (epwm)->WGCTL0 = (((epwm)->WGCTL0 & ~(3UL << (i << 1))) | ((u32ZeroLevel) << (i << 1)));
                (epwm)->WGCTL0 = (((epwm)->WGCTL0 & ~(3UL << ((16) + (i << 1)))) | ((u32PeriodLevel) << ((16) + (i << 1))));
                (epwm)->WGCTL1 = (((epwm)->WGCTL1 & ~(3UL << (i << 1))) | ((u32CmpUpLevel) << (i << 1)));
                (epwm)->WGCTL1 = (((epwm)->WGCTL1 & ~(3UL << ((16) + (i << 1)))) | ((u32CmpDownLevel) << ((16) + (i << 1))));
            }
        }
    } while (0);
}
# 584 "../../../../Library/StdDriver/inc\\epwm.h"
static __inline void EPWM_SET_DEADZONE_CLK_SRC(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32AfterPrescaler)
{
    uint32_t u32DTCLKSrc = ((u32AfterPrescaler) << (16)) << (u32ChannelNum >> 1U);
    epwm->DTCTL = ((epwm->DTCTL & ~((0x1ul << (16)) << (u32ChannelNum >> 1U))) | u32DTCLKSrc);
}




uint32_t EPWM_ConfigCaptureChannel(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32UnitTimeNsec, uint32_t u32CaptureEdge);
uint32_t EPWM_ConfigOutputChannel(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32Frequency, uint32_t u32DutyCycle);
void EPWM_Start(EPWM_T *epwm, uint32_t u32ChannelMask);
void EPWM_Stop(EPWM_T *epwm, uint32_t u32ChannelMask);
void EPWM_ForceStop(EPWM_T *epwm, uint32_t u32ChannelMask);
void EPWM_EnableADCTrigger(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32Condition);
void EPWM_DisableADCTrigger(EPWM_T *epwm, uint32_t u32ChannelNum);
int32_t EPWM_EnableADCTriggerPrescale(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32Prescale, uint32_t u32PrescaleCnt);
void EPWM_DisableADCTriggerPrescale(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_ClearADCTriggerFlag(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32Condition);
uint32_t EPWM_GetADCTriggerFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableDACTrigger(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32Condition);
void EPWM_DisableDACTrigger(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_ClearDACTriggerFlag(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32Condition);
uint32_t EPWM_GetDACTriggerFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableFaultBrake(EPWM_T *epwm, uint32_t u32ChannelMask, uint32_t u32LevelMask, uint32_t u32BrakeSource);
void EPWM_EnableCapture(EPWM_T *epwm, uint32_t u32ChannelMask);
void EPWM_DisableCapture(EPWM_T *epwm, uint32_t u32ChannelMask);
void EPWM_EnableOutput(EPWM_T *epwm, uint32_t u32ChannelMask);
void EPWM_DisableOutput(EPWM_T *epwm, uint32_t u32ChannelMask);
void EPWM_EnablePDMA(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32RisingFirst, uint32_t u32Mode);
void EPWM_DisablePDMA(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableFallingDeadZone(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32FDuration);
void EPWM_DisableFallingDeadZone(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableRisingDeadZone(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32RDuration);
void EPWM_DisableRisingDeadZone(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableCaptureInt(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32Edge);
void EPWM_DisableCaptureInt(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32Edge);
void EPWM_ClearCaptureIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32Edge);
uint32_t EPWM_GetCaptureIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableDutyInt(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32IntDutyType);
void EPWM_DisableDutyInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_ClearDutyIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
uint32_t EPWM_GetDutyIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableFaultBrakeInt(EPWM_T *epwm, uint32_t u32BrakeSource);
void EPWM_DisableFaultBrakeInt(EPWM_T *epwm, uint32_t u32BrakeSource);
void EPWM_ClearFaultBrakeIntFlag(EPWM_T *epwm, uint32_t u32BrakeSource);
uint32_t EPWM_GetFaultBrakeIntFlag(EPWM_T *epwm, uint32_t u32BrakeSource);
void EPWM_EnablePeriodInt(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32IntPeriodType);
void EPWM_DisablePeriodInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_ClearPeriodIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
uint32_t EPWM_GetPeriodIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableZeroInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_DisableZeroInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_ClearZeroIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
uint32_t EPWM_GetZeroIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableAcc(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32IntFlagCnt, uint32_t u32IntAccSrc);
void EPWM_DisableAcc(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableAccInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_DisableAccInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_ClearAccInt(EPWM_T *epwm, uint32_t u32ChannelNum);
uint32_t EPWM_GetAccInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableAccPDMA(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_DisableAccPDMA(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableAccStopMode(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_DisableAccStopMode(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_ClearFTDutyIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
uint32_t EPWM_GetFTDutyIntFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableLoadMode(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32LoadMode);
void EPWM_DisableLoadMode(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32LoadMode);
void EPWM_ConfigSyncPhase(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32SyncSrc, uint32_t u32Direction, uint32_t u32StartPhase);
void EPWM_EnableSyncPhase(EPWM_T *epwm, uint32_t u32ChannelMask);
void EPWM_DisableSyncPhase(EPWM_T *epwm, uint32_t u32ChannelMask);
void EPWM_EnableSyncNoiseFilter(EPWM_T *epwm, uint32_t u32ClkCnt, uint32_t u32ClkDivSel);
void EPWM_DisableSyncNoiseFilter(EPWM_T *epwm);
void EPWM_EnableSyncPinInverse(EPWM_T *epwm);
void EPWM_DisableSyncPinInverse(EPWM_T *epwm);
void EPWM_SetClockSource(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32ClkSrcSel);
void EPWM_EnableBrakeNoiseFilter(EPWM_T *epwm, uint32_t u32BrakePinNum, uint32_t u32ClkCnt, uint32_t u32ClkDivSel);
void EPWM_DisableBrakeNoiseFilter(EPWM_T *epwm, uint32_t u32BrakePinNum);
void EPWM_EnableBrakePinInverse(EPWM_T *epwm, uint32_t u32BrakePinNum);
void EPWM_DisableBrakePinInverse(EPWM_T *epwm, uint32_t u32BrakePinNum);
void EPWM_SetBrakePinSource(EPWM_T *epwm, uint32_t u32BrakePinNum, uint32_t u32SelAnotherModule);
void EPWM_SetLeadingEdgeBlanking(EPWM_T *epwm, uint32_t u32TrigSrcSel, uint32_t u32TrigType, uint32_t u32BlankingCnt, uint32_t u32BlankingEnable);
uint32_t EPWM_GetWrapAroundFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_ClearWrapAroundFlag(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableFaultDetect(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32AfterPrescaler, uint32_t u32ClkSel);
void EPWM_DisableFaultDetect(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableFaultDetectOutput(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_DisableFaultDetectOutput(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableFaultDetectDeglitch(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32DeglitchSmpCycle);
void EPWM_DisableFaultDetectDeglitch(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableFaultDetectMask(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32MaskCnt);
void EPWM_DisableFaultDetectMask(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableFaultDetectInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_DisableFaultDetectInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_ClearFaultDetectInt(EPWM_T *epwm, uint32_t u32ChannelNum);
uint32_t EPWM_GetFaultDetectInt(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableCaptureInputNoiseFilter(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32FilterCount, uint32_t u32ClkSrcSel);
void EPWM_DisableCaptureInputNoiseFilter(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableExtEventTrigger(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32ExtEventSrc, uint32_t u32CounterAction);
void EPWM_DisableExtEventTrigger(EPWM_T *epwm, uint32_t u32ChannelNum);
uint32_t EPWM_GetAccCounter(EPWM_T *epwm, uint32_t u32ChannelNum);
void EPWM_EnableSWEventOutput(EPWM_T *epwm, uint32_t u32ChannelNum, uint32_t u32OutputLevel);
void EPWM_DisableSWEventOutput(EPWM_T *epwm, uint32_t u32ChannelNum);
# 1698 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\eqei.h" 1
# 426 "../../../../Library/StdDriver/inc\\eqei.h"
static __inline void EQEI_StartUintTimer(EQEI_T *eqei);
static __inline void EQEI_StopUintTimer(EQEI_T *eqei);
static __inline void EQEI_EnableUintTimerINT(EQEI_T *eqei);
static __inline void EQEI_DisableUintTimerINT(EQEI_T *eqei);
# 440 "../../../../Library/StdDriver/inc\\eqei.h"
static __inline void EQEI_StartUintTimer(EQEI_T *eqei)
{
    eqei->CTL2 |= (0x1ul << (8));
}
# 454 "../../../../Library/StdDriver/inc\\eqei.h"
static __inline void EQEI_StopUintTimer(EQEI_T *eqei)
{
    eqei->CTL2 &= ~(0x1ul << (8));
}
# 468 "../../../../Library/StdDriver/inc\\eqei.h"
static __inline void EQEI_EnableUintTimerINT(EQEI_T *eqei)
{
    eqei->CTL2 |= (0x1ul << (17));
}
# 482 "../../../../Library/StdDriver/inc\\eqei.h"
static __inline void EQEI_DisableUintTimerINT(EQEI_T *eqei)
{
    eqei->CTL2 &= ~(0x1ul << (17));
}

void EQEI_Close(EQEI_T *eqei);
void EQEI_DisableInt(EQEI_T *eqei, uint32_t u32IntSel);
void EQEI_EnableInt(EQEI_T *eqei, uint32_t u32IntSel);
void EQEI_Open(EQEI_T *eqei, uint32_t u32Mode, uint32_t u32Value);
void EQEI_Start(EQEI_T *eqei);
void EQEI_Stop(EQEI_T *eqei);
# 1699 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\fmc.h" 1
# 177 "../../../../Library/StdDriver/inc\\fmc.h"
extern int32_t g_FMC_i32ErrCode;
# 190 "../../../../Library/StdDriver/inc\\fmc.h"
static __inline void FMC_Open(void)
{
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPCTL |= (0x1ul << (0));
}




static __inline void FMC_Close(void)
{
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPCTL &= ~(0x1ul << (0));
}
# 212 "../../../../Library/StdDriver/inc\\fmc.h"
static __inline uint32_t FMC_ReadCID(void)
{
    int32_t i32TimeOutCnt = (SystemCoreClock >> 3);

    g_FMC_i32ErrCode = 0;

    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPCMD = 0x0BUL;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPADDR = 0x0u;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG = (0x1ul << (0));

    while (((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG & (0x1ul << (0)))
    {
        if (i32TimeOutCnt-- <= 0)
        {
            g_FMC_i32ErrCode = (-1UL);
            return 0xFFFFFFFF;
        }
    }

    return ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPDAT;
}
# 243 "../../../../Library/StdDriver/inc\\fmc.h"
static __inline uint32_t FMC_ReadPID(void)
{
    int32_t i32TimeOutCnt = (SystemCoreClock >> 3);

    g_FMC_i32ErrCode = 0;

    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPCMD = 0x0CUL;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPADDR = 0x04u;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG = (0x1ul << (0));

    while (((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG & (0x1ul << (0)))
    {
        if (i32TimeOutCnt-- <= 0)
        {
            g_FMC_i32ErrCode = (-1UL);
            return 0xFFFFFFFF;
        }
    }

    return ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPDAT;
}
# 274 "../../../../Library/StdDriver/inc\\fmc.h"
static __inline uint32_t FMC_ReadUID(uint8_t u8Index)
{
    int32_t i32TimeOutCnt = (SystemCoreClock >> 3);

    g_FMC_i32ErrCode = 0;

    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPCMD = 0x04UL;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPADDR = ((uint32_t)u8Index << 2u);
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPDAT = 0u;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG = 0x1u;

    while (((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG & (0x1ul << (0)))
    {
        if (i32TimeOutCnt-- <= 0)
        {
            g_FMC_i32ErrCode = (-1UL);
            return 0xFFFFFFFF;
        }
    }

    return ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPDAT;
}
# 306 "../../../../Library/StdDriver/inc\\fmc.h"
static __inline uint32_t FMC_ReadUCID(uint32_t u32Index)
{
    int32_t i32TimeOutCnt = (SystemCoreClock >> 3);

    g_FMC_i32ErrCode = 0;

    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPCMD = 0x04UL;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPADDR = (0x04u * u32Index) + 0x10u;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG = (0x1ul << (0));

    while (((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG & (0x1ul << (0)))
    {
        if (i32TimeOutCnt-- <= 0)
        {
            g_FMC_i32ErrCode = (-1UL);
            return 0xFFFFFFFF;
        }
    }

    return ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPDAT;
}
# 341 "../../../../Library/StdDriver/inc\\fmc.h"
static __inline int32_t FMC_SetVectorPageAddr(uint32_t u32PageAddr)
{
    int32_t i32TimeOutCnt = (SystemCoreClock >> 3);

    g_FMC_i32ErrCode = 0;

    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPCMD = 0x2EUL;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPADDR = u32PageAddr;
    ((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG = 0x1u;

    while (((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPTRG)
    {
        if (i32TimeOutCnt-- <= 0)
        {
            g_FMC_i32ErrCode = (-1UL);
            return (-1UL);
        }
    }

    return 0;
}
# 372 "../../../../Library/StdDriver/inc\\fmc.h"
static __inline uint32_t FMC_GetVECMAP(void)
{
    return (((FMC_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x44000UL))->ISPSTS & (0x7fffful << (9)));
}




extern uint32_t FMC_Read(uint32_t u32Addr);
extern int32_t FMC_Read_64(uint32_t u32Addr, uint32_t *pu32Data0, uint32_t *pu32Data1);
extern int32_t FMC_Write(uint32_t u32Addr, uint32_t u32Data);
extern int32_t FMC_Write8Bytes(uint32_t u32Addr, uint32_t u32Data0, uint32_t u32Data1);
extern int32_t FMC_WriteMultiple(uint32_t u32Addr, uint32_t pu32Buf[], uint32_t u32ByteLen);
extern int32_t FMC_Erase(uint32_t u32PageAddr);
extern int32_t FMC_ReadConfig(uint32_t u32Config[], uint32_t u32Count);
extern int32_t FMC_WriteConfig(uint32_t u32ConfigAddr, uint32_t u32ConfigVal);
extern int32_t FMC_EraseConfig(uint32_t u32ConfigAddr);
extern int32_t FMC_RemapBank(uint32_t u32Bank);
extern int32_t FMC_Erase_Bank(uint32_t u32BankAddr);
extern int32_t FMC_ConfigNPUXOM(uint32_t u32XomBase, uint8_t u8XomPageCnt, uint32_t u32DRBound);
extern int32_t FMC_ConfigXOM(uint32_t u32XomNum, uint32_t u32XomBase, uint8_t u8XomPageCnt);
extern int32_t FMC_GetXOMState(uint32_t u32XomNum);
extern int32_t FMC_EraseXOM(uint32_t u32XomNum);
extern int32_t FMC_GetBootSource(void);
extern int32_t FMC_ReadOTP(uint32_t u32OtpNum, uint32_t *pu32LowWord, uint32_t *pu32HighWord);
extern int32_t FMC_WriteOTP(uint32_t u32OtpNum, uint32_t u32LowWord, uint32_t u32HighWord);
extern int32_t FMC_LockOTP(uint32_t u32OtpNum);
extern int32_t FMC_IsOTPLocked(uint32_t u32OtpNum);
extern int32_t FMC_ConfigSecureConceal(uint32_t u32Base, uint32_t u32PageCnt, uint32_t bActiveEnable);
# 1700 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h" 1
# 11 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
# 1 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h" 1
# 20 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
# 1 "../../../../Library/StdDriver/inc\\gdma\\dma350_regdef.h" 1
# 27 "../../../../Library/StdDriver/inc\\gdma\\dma350_regdef.h"
typedef struct
{
    volatile uint32_t CH_CMD;
    volatile uint32_t CH_STATUS;
    volatile uint32_t CH_INTREN;
    volatile uint32_t CH_CTRL;
    volatile uint32_t CH_SRCADDR;
    volatile uint32_t CH_SRCADDRHI;
    volatile uint32_t CH_DESADDR;
    volatile uint32_t CH_DESADDRHI;
    volatile uint32_t CH_XSIZE;
    volatile uint32_t CH_XSIZEHI;
    volatile uint32_t CH_SRCTRANSCFG;
    volatile uint32_t CH_DESTRANSCFG;
    volatile uint32_t CH_XADDRINC;
    volatile uint32_t CH_YADDRSTRIDE;
    volatile uint32_t CH_FILLVAL;
    volatile uint32_t CH_YSIZE;
    volatile uint32_t CH_TMPLTCFG;
    volatile uint32_t CH_SRCTMPLT;
    volatile uint32_t CH_DESTMPLT;
    volatile uint32_t CH_SRCTRIGINCFG;
    volatile uint32_t CH_DESTRIGINCFG;
    volatile uint32_t CH_TRIGOUTCFG;
    volatile uint32_t CH_GPOEN0;
    uint32_t RESERVED0[1];
    volatile uint32_t CH_GPOVAL0;
    uint32_t RESERVED1[1];
    volatile uint32_t CH_STREAMINTCFG;
    uint32_t RESERVED2[1];
    volatile uint32_t CH_LINKATTR;
    volatile uint32_t CH_AUTOCFG;
    volatile uint32_t CH_LINKADDR;
    volatile uint32_t CH_LINKADDRHI;
    volatile uint32_t CH_GPOREAD0;
    uint32_t RESERVED3[1];
    volatile uint32_t CH_WRKREGPTR;
    volatile uint32_t CH_WRKREGVAL;
    volatile uint32_t CH_ERRINFO;
    uint32_t RESERVED4[13];
    volatile uint32_t CH_IIDR;
    volatile uint32_t CH_AIDR;
    uint32_t RESERVED5[6];
    volatile uint32_t CH_ISSUECAP;
    uint32_t RESERVED6[3];
    volatile uint32_t CH_BUILDCFG0;
    volatile uint32_t CH_BUILDCFG1;
} DMACH_TypeDef;
typedef struct
{
    volatile uint32_t NSEC_CHINTRSTATUS0;
    uint32_t RESERVED0[1];
    volatile uint32_t NSEC_STATUS;
    volatile uint32_t NSEC_CTRL;
    uint32_t RESERVED1[1];
    volatile uint32_t NSEC_CHPTR;
    volatile uint32_t NSEC_CHCFG;
    uint32_t RESERVED2[53];
    volatile uint32_t NSEC_STATUSPTR;
    volatile uint32_t NSEC_STATUSVAL;
    volatile uint32_t NSEC_SIGNALPTR;
    volatile uint32_t NSEC_SIGNALVAL;
} DMANSECCTRL_TypeDef;
typedef struct
{
    volatile uint32_t SEC_CHINTRSTATUS0;
    uint32_t RESERVED0[1];
    volatile uint32_t SEC_STATUS;
    volatile uint32_t SEC_CTRL;
    uint32_t RESERVED1[1];
    volatile uint32_t SEC_CHPTR;
    volatile uint32_t SEC_CHCFG;
    uint32_t RESERVED2[53];
    volatile uint32_t SEC_STATUSPTR;
    volatile uint32_t SEC_STATUSVAL;
    volatile uint32_t SEC_SIGNALPTR;
    volatile uint32_t SEC_SIGNALVAL;
} DMASECCTRL_TypeDef;
typedef struct
{
    volatile uint32_t SCFG_CHSEC0;
    uint32_t RESERVED0[1];
    volatile uint32_t SCFG_TRIGINSEC0;
    uint32_t RESERVED1[7];
    volatile uint32_t SCFG_TRIGOUTSEC0;
    uint32_t RESERVED2[5];
    volatile uint32_t SCFG_CTRL;
    volatile uint32_t SCFG_INTRSTATUS;
} DMASECCFG_TypeDef;
typedef struct
{
    uint32_t RESERVED0[44];
    volatile uint32_t DMA_BUILDCFG0;
    volatile uint32_t DMA_BUILDCFG1;
    volatile uint32_t DMA_BUILDCFG2;
    uint32_t RESERVED1[3];
    volatile uint32_t IIDR;
    volatile uint32_t AIDR;
    volatile uint32_t PIDR4;
    uint32_t RESERVED2[3];
    volatile uint32_t PIDR0;
    volatile uint32_t PIDR1;
    volatile uint32_t PIDR2;
    volatile uint32_t PIDR3;
    volatile uint32_t CIDR0;
    volatile uint32_t CIDR1;
    volatile uint32_t CIDR2;
    volatile uint32_t CIDR3;
} DMAINFO_TypeDef;
# 21 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h" 2




# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stdbool.h" 1 3
# 26 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h" 2
# 90 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
enum dma350_ch_error_t
{
    DMA350_CH_ERR_NONE = 0,
    DMA350_CH_ERR_IIDR_MISMATCH,

    DMA350_CH_ERR_AIDR_MISMATCH,

    DMA350_CH_ERR_INVALID_ARG,
    DMA350_CH_ERR_INVALID_CMD,
    DMA350_CH_ERR_NOT_READY,
    DMA350_CH_ERR_UNSUP_CH,
};


struct dma350_ch_dev_cfg_t
{
    DMACH_TypeDef *const ch_base;
    const uint8_t channel;
};


struct dma350_ch_dev_data_t
{
    uint32_t state;

};


struct dma350_ch_dev_t
{
    const struct dma350_ch_dev_cfg_t cfg;
    struct dma350_ch_dev_data_t data;
};


enum dma350_ch_cmd_t
{
    DMA350_CH_CMD_ENABLECMD = (0x1UL << (0U)),
    DMA350_CH_CMD_CLEARCMD = (0x1UL << (1U)),
    DMA350_CH_CMD_DISABLECMD = (0x1UL << (2U)),
    DMA350_CH_CMD_STOPCMD = (0x1UL << (3U)),
    DMA350_CH_CMD_PAUSECMD = (0x1UL << (4U)),
    DMA350_CH_CMD_RESUMECMD = (0x1UL << (5U)),
    DMA350_CH_CMD_SRCSWTRIGINREQ = (0x1UL << (16U)),
    DMA350_CH_CMD_SRCSWTRIGINREQ_LAST =
        (0x1UL << (16U)) | (0x1UL << (17U)),
    DMA350_CH_CMD_SRCSWTRIGINREQ_BLOCK =
        (0x1UL << (16U)) | (0x2UL << (17U)),
    DMA350_CH_CMD_SRCSWTRIGINREQ_BLOCK_LAST = (0x1UL << (16U)) |
                                              (0x2UL << (17U)) |
                                              (0x1UL << (17U)),
    DMA350_CH_CMD_DESSWTRIGINREQ = (0x1UL << (20U)),
    DMA350_CH_CMD_DESSWTRIGINREQ_LAST =
        (0x1UL << (20U)) | (0x1UL << (21U)),
    DMA350_CH_CMD_DESSWTRIGINREQ_BLOCK =
        (0x1UL << (20U)) | (0x2UL << (21U)),
    DMA350_CH_CMD_DESSWTRIGINREQ_BLOCK_LAST = (0x1UL << (20U)) |
                                              (0x2UL << (21U)) |
                                              (0x1UL << (21U)),
    DMA350_CH_CMD_SWTRIGOUTACK = (0x1UL << (24U))
};


enum dma350_ch_stat_t
{
    DMA350_CH_STAT_DONE = (0x1UL << (16U)),
    DMA350_CH_STAT_ERR = (0x1UL << (17U)),
    DMA350_CH_STAT_DISABLED = (0x1UL << (18U)),
    DMA350_CH_STAT_STOPPED = (0x1UL << (19U)),
    DMA350_CH_STAT_SRCTRIGINWAIT = (0x1UL << (24U)),
    DMA350_CH_STAT_DESTRIGINWAIT = (0x1UL << (25U)),
    DMA350_CH_STAT_TRIGOUTACKWAIT = (0x1UL << (26U)),
    DMA350_CH_STAT_ALL =
        (0x1UL << (16U)) | (0x1UL << (17U)) |
        (0x1UL << (18U)) | (0x1UL << (19U)) |
        (0x1UL << (24U)) | (0x1UL << (25U)) |
        (0x1UL << (26U))
};


enum dma350_ch_intr_t
{
    DMA350_CH_INTREN_DONE = (0x1UL << (0U)),
    DMA350_CH_INTREN_ERR = (0x1UL << (1U)),
    DMA350_CH_INTREN_DISABLED = (0x1UL << (2U)),
    DMA350_CH_INTREN_STOPPED = (0x1UL << (3U)),
    DMA350_CH_INTREN_SRCTRIGINWAIT = (0x1UL << (8U)),
    DMA350_CH_INTREN_DESTRIGINWAIT = (0x1UL << (9U)),
    DMA350_CH_INTREN_TRIGOUTACKWAIT = (0x1UL << (10U)),
    DMA350_CH_INTREN_ALL =
        (0x1UL << (0U)) | (0x1UL << (1U)) |
        (0x1UL << (2U)) | (0x1UL << (3U)) |
        (0x1UL << (8U)) |
        (0x1UL << (9U)) | (0x1UL << (10U))
};


enum dma350_ch_transize_t
{
    DMA350_CH_TRANSIZE_8BITS = 0,
    DMA350_CH_TRANSIZE_16BITS = (0x1UL << (0U)),
    DMA350_CH_TRANSIZE_32BITS = (0x2UL << (0U)),
    DMA350_CH_TRANSIZE_64BITS = (0x2UL << (0U)) | (0x1UL << (0U)),
    DMA350_CH_TRANSIZE_128BITS = (0x4UL << (0U)),
    DMA350_CH_TRANSIZE_256BITS =
        (0x4UL << (0U)) | (0x1UL << (0U)),
    DMA350_CH_TRANSIZE_512BITS =
        (0x4UL << (0U)) | (0x2UL << (0U)),
    DMA350_CH_TRANSIZE_1024BITS =
        (0x4UL << (0U)) | (0x2UL << (0U)) | (0x1UL << (0U))
};


enum dma350_ch_xtype_t
{
    DMA350_CH_XTYPE_DISABLE = 0,
    DMA350_CH_XTYPE_CONTINUE = (0x1UL << (9U)),
    DMA350_CH_XTYPE_WRAP = (0x2UL << (9U)),
    DMA350_CH_XTYPE_FILL = (0x2UL << (9U)) | (0x1UL << (9U))
};


enum dma350_ch_ytype_t
{
    DMA350_CH_YTYPE_DISABLE = 0,
    DMA350_CH_YTYPE_CONTINUE = (0x1UL << (12U)),
    DMA350_CH_YTYPE_WRAP = (0x2UL << (12U)),
    DMA350_CH_YTYPE_FILL = (0x2UL << (12U)) | (0x1UL << (12U))
};


enum dma350_ch_regreloadtype_t
{
    DMA350_CH_REGRELOADTYPE_DISABLE = 0,
    DMA350_CH_REGRELOADTYPE_SRC_DES_SIZE = (0x1UL << (18U)),
    DMA350_CH_REGRELOADTYPE_SRC_ADDR_SRC_DES_SIZE =
        (0x2UL << (18U)) | (0x1UL << (18U)),
    DMA350_CH_REGRELOADTYPE_DES_ADDR_SRC_DES_SIZE =
        (0x4UL << (18U)) | (0x1UL << (18U)),
    DMA350_CH_REGRELOADTYPE_SRC_DES_ADDR_AND_SIZE =
        (0x4UL << (18U)) | (0x2UL << (18U)) |
        (0x1UL << (18U))
};


enum dma350_ch_donetype_t
{
    DMA350_CH_DONETYPE_NONE = 0,
    DMA350_CH_DONETYPE_END_OF_CMD = (0x1UL << (21U)),
    DMA350_CH_DONETYPE_END_OF_AUTORESTART =
        (0x2UL << (21U)) | (0x1UL << (21U))
};


enum dma350_ch_srctrigintype_t
{
    DMA350_CH_SRCTRIGINTYPE_SOFTWARE_ONLY = 0,
    DMA350_CH_SRCTRIGINTYPE_HW = (0x2UL << (8U)),
    DMA350_CH_SRCTRIGINTYPE_INTERNAL = (0x2UL << (8U)) |
                                       (0x1UL << (8U))
};


enum dma350_ch_srctriginmode_t
{
    DMA350_CH_SRCTRIGINMODE_CMD = 0,
    DMA350_CH_SRCTRIGINMODE_DMA_FLOW_CTRL = (0x2UL << (10U)),
    DMA350_CH_SRCTRIGINMODE_PERIPH_FLOW_CTRL =
        (0x2UL << (10U)) |
        (0x1UL << (10U))
};


enum dma350_ch_destrigintype_t
{
    DMA350_CH_DESTRIGINTYPE_SOFTWARE_ONLY = 0,
    DMA350_CH_DESTRIGINTYPE_HW = (0x2UL << (8U)),
    DMA350_CH_DESTRIGINTYPE_INTERNAL = (0x2UL << (8U)) |
                                       (0x1UL << (8U))
};


enum dma350_ch_destriginmode_t
{
    DMA350_CH_DESTRIGINMODE_CMD = 0,
    DMA350_CH_DESTRIGINMODE_DMA_FLOW_CTRL = (0x2UL << (10U)),
    DMA350_CH_DESTRIGINMODE_PERIPH_FLOW_CTRL =
        (0x2UL << (10U)) |
        (0x1UL << (10U))
};


enum dma350_ch_trigouttype_t
{
    DMA350_CH_TRIGOUTTYPE_SOFTWARE_ONLY = 0,
    DMA350_CH_TRIGOUTTYPE_HW = (0x2UL << (8U)),
    DMA350_CH_TRIGOUTTYPE_INTERNAL =
        (0x2UL << (8U)) | (0x1UL << (8U))
};


enum dma350_ch_streamtype_t
{
    DMA350_CH_STREAMTYPE_IN_OUT = 0,
    DMA350_CH_STREAMTYPE_OUT_ONLY = (0x1UL << (9U)),
    DMA350_CH_STREAMTYPE_IN_ONLY = (0x2UL << (9U))
};


union dma350_ch_status_t
{
    struct
    {
        uint32_t INTR_DONE: 1;
        uint32_t INTR_ERR: 1;
        uint32_t INTR_DISABLED: 1;
        uint32_t INTR_STOPPED: 1;
        uint32_t RESERVED0: 4;
        uint32_t INTR_SRCTRIGINWAIT: 1;
        uint32_t INTR_DESTRIGINWAIT: 1;
        uint32_t INTR_TRIGOUTACKWAIT: 1;
        uint32_t RESERVED1: 5;
        uint32_t STAT_DONE: 1;
        uint32_t STAT_ERR: 1;
        uint32_t STAT_DISABLED: 1;
        uint32_t STAT_STOPPED: 1;
        uint32_t STAT_PAUSED: 1;
        uint32_t STAT_RESUMEWAIT: 1;
        uint32_t RESERVED2: 2;
        uint32_t STAT_SRCTRIGINWAIT: 1;
        uint32_t STAT_DESTRIGINWAIT: 1;
        uint32_t STAT_TRIGOUTACKWAIT: 1;
        uint32_t RESERVED3: 5;
    } b;
    uint32_t w;
};



struct dma350_cmdlink_reg_t
{

    uint32_t intren;
    uint32_t ctrl;
    uint32_t srcaddr;
    uint32_t srcaddrhi;
    uint32_t desaddr;
    uint32_t desaddrhi;
    uint32_t xsize;
    uint32_t xsizehi;
    uint32_t srctranscfg;
    uint32_t destranscfg;
    uint32_t xaddrinc;
    uint32_t yaddrstride;
    uint32_t fillval;
    uint32_t ysize;
    uint32_t tmpltcfg;
    uint32_t srctmplt;
    uint32_t destmplt;
    uint32_t srctrigincfg;
    uint32_t destrigincfg;
    uint32_t trigoutcfg;
    uint32_t gpoen0;
    uint32_t reserved0;
    uint32_t gpoval0;
    uint32_t reserved1;
    uint32_t streamintcfg;
    uint32_t reserved2;
    uint32_t linkattr;
    uint32_t autocfg;
    uint32_t linkaddr;
    uint32_t linkaddrhi;
};


struct dma350_cmdlink_gencfg_t
{
    uint32_t header;
    struct dma350_cmdlink_reg_t cfg;
};
# 380 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
enum dma350_ch_error_t dma350_ch_init(struct dma350_ch_dev_t *dev);
# 391 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
_Bool dma350_ch_is_init(const struct dma350_ch_dev_t *dev);
# 404 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_src(struct dma350_ch_dev_t *dev, uint32_t src_addr);
# 417 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_des(struct dma350_ch_dev_t *dev, uint32_t des_addr);
# 428 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_chprio(struct dma350_ch_dev_t *dev, uint8_t chprio);
# 442 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_xsize16(struct dma350_ch_dev_t *dev, uint16_t src_xsize,
                           uint16_t des_xsize);
# 457 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_xsize32(struct dma350_ch_dev_t *dev, uint32_t src_xsize,
                           uint32_t des_xsize);
# 472 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_ysize16(struct dma350_ch_dev_t *dev, uint16_t src_ysize,
                           uint16_t des_ysize);
# 487 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_yaddrstride(struct dma350_ch_dev_t *dev,
                               uint16_t src_yaddrstride,
                               uint16_t des_yaddrstride);
# 502 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_transize(struct dma350_ch_dev_t *dev,
                            enum dma350_ch_transize_t transize);
# 517 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_xtype(struct dma350_ch_dev_t *dev,
                         enum dma350_ch_xtype_t xtype);
# 533 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_ytype(struct dma350_ch_dev_t *dev,
                         enum dma350_ch_ytype_t ytype);
# 547 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_regreloadtype(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_regreloadtype_t regreloadtype);
# 561 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_donetype(struct dma350_ch_dev_t *dev,
                            enum dma350_ch_donetype_t donetype);
# 574 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_enable_donepause(struct dma350_ch_dev_t *dev);
# 586 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_disable_donepause(struct dma350_ch_dev_t *dev);
# 598 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_enable_srctrigin(struct dma350_ch_dev_t *dev);
# 610 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_disable_srctrigin(struct dma350_ch_dev_t *dev);
# 622 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_enable_destrigin(struct dma350_ch_dev_t *dev);
# 634 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_disable_destrigin(struct dma350_ch_dev_t *dev);
# 646 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_enable_trigout(struct dma350_ch_dev_t *dev);
# 658 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_disable_trigout(struct dma350_ch_dev_t *dev);
# 670 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_enable_gpo(struct dma350_ch_dev_t *dev);
# 682 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_disable_gpo(struct dma350_ch_dev_t *dev);
# 694 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_enable_stream(struct dma350_ch_dev_t *dev);
# 706 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_disable_stream(struct dma350_ch_dev_t *dev);
# 718 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_srcmemattr(struct dma350_ch_dev_t *dev, uint8_t memattr,
                              uint8_t shareattr);
# 730 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_srcmemattrlo(struct dma350_ch_dev_t *dev, uint8_t memattrlo);
# 741 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_srcmemattrhi(struct dma350_ch_dev_t *dev, uint8_t memattrhi);
# 752 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_srcshareattr(struct dma350_ch_dev_t *dev, uint8_t shareattr);
# 764 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_desmemattr(struct dma350_ch_dev_t *dev, uint8_t memattr,
                              uint8_t shareattr);
# 776 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_desmemattrlo(struct dma350_ch_dev_t *dev, uint8_t memattrlo);
# 787 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_desmemattrhi(struct dma350_ch_dev_t *dev, uint8_t memattrhi);
# 798 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_desshareattr(struct dma350_ch_dev_t *dev, uint8_t shareattr);
# 810 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_src_trans_secure(struct dma350_ch_dev_t *dev);
# 822 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_src_trans_nonsecure(struct dma350_ch_dev_t *dev);
# 834 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_des_trans_secure(struct dma350_ch_dev_t *dev);
# 846 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_des_trans_nonsecure(struct dma350_ch_dev_t *dev);
# 858 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_src_trans_privileged(struct dma350_ch_dev_t *dev);
# 870 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_src_trans_unprivileged(struct dma350_ch_dev_t *dev);
# 882 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_des_trans_privileged(struct dma350_ch_dev_t *dev);
# 894 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_des_trans_unprivileged(struct dma350_ch_dev_t *dev);
# 905 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_srcmaxburstlen(struct dma350_ch_dev_t *dev, uint8_t length);
# 916 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_desmaxburstlen(struct dma350_ch_dev_t *dev, uint8_t length);
# 929 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_src_xaddr_inc(struct dma350_ch_dev_t *dev,
                                 int16_t src_xaddr_inc);
# 943 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_des_xaddr_inc(struct dma350_ch_dev_t *dev,
                                 int16_t des_xaddr_inc);
# 958 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_xaddr_inc(struct dma350_ch_dev_t *dev, int16_t src_xaddr_inc,
                             int16_t des_xaddr_inc);
# 972 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_fill_value(struct dma350_ch_dev_t *dev, uint32_t fill_value);
# 985 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_cmd(struct dma350_ch_dev_t *dev, enum dma350_ch_cmd_t cmd);
# 997 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
enum dma350_ch_cmd_t dma350_ch_get_cmd(struct dma350_ch_dev_t *dev);
# 1010 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_enable_intr(struct dma350_ch_dev_t *dev,
                           enum dma350_ch_intr_t intr);
# 1024 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_disable_intr(struct dma350_ch_dev_t *dev,
                            enum dma350_ch_intr_t intr);
# 1039 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
_Bool dma350_ch_is_intr_set(struct dma350_ch_dev_t *dev,
                           enum dma350_ch_intr_t intr);
# 1055 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
_Bool dma350_ch_is_stat_set(struct dma350_ch_dev_t *dev,
                           enum dma350_ch_stat_t stat);
# 1069 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_clear_stat(struct dma350_ch_dev_t *dev,
                          enum dma350_ch_stat_t stat);
# 1084 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_autocfg_restart_cnt(struct dma350_ch_dev_t *dev,
                                       uint16_t cnt);
# 1099 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_autocfg_restart_inf(struct dma350_ch_dev_t *dev);
# 1112 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_tmplt_src_size(struct dma350_ch_dev_t *dev, uint32_t size);
# 1125 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_tmplt_des_size(struct dma350_ch_dev_t *dev, uint32_t size);
# 1139 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_tmplt_src(struct dma350_ch_dev_t *dev, uint32_t tmplt);
# 1153 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_tmplt_des(struct dma350_ch_dev_t *dev, uint32_t tmplt);
# 1165 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
uint32_t dma350_ch_get_tmplt_src(struct dma350_ch_dev_t *dev);
# 1177 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
uint32_t dma350_ch_get_tmplt_des(struct dma350_ch_dev_t *dev);
# 1188 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_srctriginsel(struct dma350_ch_dev_t *dev,
                                uint8_t srctriginsel);
# 1200 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_srctrigintype(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_srctrigintype_t type);
# 1212 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_srctriginmode(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_srctriginmode_t mode);
# 1224 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_srctriginblksize(struct dma350_ch_dev_t *dev,
                                    uint8_t blksize);
# 1236 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_destriginsel(struct dma350_ch_dev_t *dev,
                                uint8_t destriginsel);
# 1248 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_destrigintype(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_destrigintype_t type);
# 1260 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_destriginmode(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_destriginmode_t mode);
# 1273 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_destriginblksize(struct dma350_ch_dev_t *dev,
                                    uint8_t blksize);
# 1285 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_trigoutsel(struct dma350_ch_dev_t *dev, uint8_t trigoutsel);
# 1296 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_trigouttype(struct dma350_ch_dev_t *dev,
                               enum dma350_ch_trigouttype_t type);
# 1309 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_gpoen0(struct dma350_ch_dev_t *dev, uint32_t gpoen0);
# 1320 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_gpoval0(struct dma350_ch_dev_t *dev, uint32_t gpoval0);
# 1330 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
uint32_t dma350_ch_get_gpoval0(struct dma350_ch_dev_t *dev);
# 1341 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_streamtype(struct dma350_ch_dev_t *dev,
                              enum dma350_ch_streamtype_t type);
# 1353 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_linkmemattrlo(struct dma350_ch_dev_t *dev,
                                 uint8_t memattrlo);
# 1365 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_linkmemattrhi(struct dma350_ch_dev_t *dev,
                                 uint8_t memattrhi);
# 1377 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_linkshareattr(struct dma350_ch_dev_t *dev,
                                 uint8_t shareattr);
# 1388 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_enable_linkaddr(struct dma350_ch_dev_t *dev);
# 1398 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_disable_linkaddr(struct dma350_ch_dev_t *dev);
# 1409 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_ch_set_linkaddr32(struct dma350_ch_dev_t *dev, uint32_t linkaddr);
# 1421 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
_Bool dma350_ch_is_busy(struct dma350_ch_dev_t *dev);
# 1433 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
_Bool dma350_ch_is_ready(struct dma350_ch_dev_t *dev);
# 1445 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
union dma350_ch_status_t dma350_ch_get_status(struct dma350_ch_dev_t *dev);
# 1457 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
union dma350_ch_status_t dma350_ch_wait_status(struct dma350_ch_dev_t *dev);
# 1468 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_regclear(struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1480 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_enable_intr(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                enum dma350_ch_intr_t intr);
# 1493 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_disable_intr(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 enum dma350_ch_intr_t intr);
# 1506 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_transize(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 enum dma350_ch_transize_t transize);
# 1519 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_chprio(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                               uint8_t chprio);
# 1532 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_xtype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                              enum dma350_ch_xtype_t xtype);
# 1545 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_ytype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                              enum dma350_ch_ytype_t ytype);
# 1558 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_regreloadtype(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_regreloadtype_t regreloadtype);
# 1573 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_donetype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 enum dma350_ch_donetype_t donetype);
# 1586 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_enable_donepause(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1599 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_disable_donepause(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1611 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_enable_srctrigin(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1623 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_disable_srctrigin(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1635 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_enable_destrigin(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1647 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_disable_destrigin(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1659 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_enable_trigout(struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1670 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_disable_trigout(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1682 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_enable_gpo(struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1693 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_disable_gpo(struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1704 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_enable_stream(struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1715 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_disable_stream(struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1727 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srcaddr32(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                  uint32_t src_addr);
# 1740 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_desaddr32(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                  uint32_t des_addr);
# 1755 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_xsize16(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint16_t src_xsize, uint16_t des_xsize);
# 1770 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_xsize32(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint32_t src_xsize, uint32_t des_xsize);
# 1784 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srcmemattrlo(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrlo);
# 1798 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srcmemattrhi(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrhi);
# 1811 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srcshareattr(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t shareattr);
# 1825 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_desmemattrlo(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrlo);
# 1839 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_desmemattrhi(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrhi);
# 1853 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_desshareattr(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t shareattr);
# 1865 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_src_trans_secure(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1877 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_src_trans_nonsecure(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1889 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_des_trans_secure(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1902 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_des_trans_nonsecure(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1915 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_src_trans_privileged(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1928 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_src_trans_unprivileged(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1941 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_des_trans_privileged(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1954 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_des_trans_unprivileged(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 1967 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srcmaxburstlen(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t length);
# 1980 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_desmaxburstlen(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t length);
# 1996 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_xaddrinc(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 uint16_t src_xaddrinc, uint16_t des_xaddrinc);
# 2012 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_yaddrstride(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                    uint16_t src_yaddrstride,
                                    uint16_t des_yaddrstride);
# 2027 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_fillval(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint32_t fillval);
# 2042 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_ysize16(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint16_t src_ysize, uint16_t des_ysize);
# 2056 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_tmpltsize(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                  uint8_t scr_tmpltsize, uint8_t des_tmpltsize);
# 2070 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srctmplt(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 uint32_t src_tmplt);
# 2084 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_destmplt(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 uint32_t des_tmplt);
# 2097 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srctriginsel(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t srctriginsel);
# 2110 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srctrigintype(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_srctrigintype_t type);
# 2124 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srctriginmode(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_srctriginmode_t mode);
# 2139 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_srctriginblksize(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t blksize);
# 2152 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_destriginsel(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t destriginsel);
# 2165 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_destrigintype(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_destrigintype_t type);
# 2179 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_destriginmode(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_destriginmode_t mode);
# 2194 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_destriginblksize(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t blksize);
# 2207 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_trigoutsel(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                   uint8_t trigoutsel);
# 2220 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_trigouttype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                    enum dma350_ch_trigouttype_t type);
# 2234 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_gpoen0(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                               uint32_t gpoen0);
# 2248 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_gpoval0(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint32_t gpoval0);
# 2261 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_streamtype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                   enum dma350_ch_streamtype_t type);
# 2275 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_linkmemattrlo(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrlo);
# 2289 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_linkmemattrhi(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrhi);
# 2303 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_linkshareattr(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t shareattr);
# 2316 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_cmdrestartcnt(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint16_t cnt);
# 2328 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_enable_cmdrestartinfen(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 2340 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_disable_cmdrestartinfen(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 2352 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_enable_linkaddr(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 2364 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_disable_linkaddr(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg);
# 2377 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
void dma350_cmdlink_set_linkaddr32(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                   uint32_t linkaddr);
# 2392 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
uint32_t *dma350_cmdlink_generate(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                  uint32_t *buffer, uint32_t *buffer_end);
# 2403 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
void dma350_cmdlink_init(struct dma350_cmdlink_gencfg_t *cmdlink_cfg);

static __inline
_Bool dma350_ch_is_init(const struct dma350_ch_dev_t *dev)
{
    return dev->data.state == (1UL << 0);
}

static __inline
void dma350_ch_set_src(struct dma350_ch_dev_t *dev, uint32_t src_addr)
{
    dev->cfg.ch_base->CH_SRCADDR = src_addr;
}

static __inline
void dma350_ch_set_des(struct dma350_ch_dev_t *dev, uint32_t des_addr)
{
    dev->cfg.ch_base->CH_DESADDR = des_addr;
}

static __inline
void dma350_ch_set_xsize16(struct dma350_ch_dev_t *dev, uint16_t src_xsize,
                           uint16_t des_xsize)
{
    dev->cfg.ch_base->CH_XSIZE =
        ((des_xsize & 0x0000FFFFUL) << 16U) | (src_xsize & 0x0000FFFFUL);
    dev->cfg.ch_base->CH_XSIZEHI = 0;
}

static __inline
void dma350_ch_set_xsize32(struct dma350_ch_dev_t *dev, uint32_t src_xsize,
                           uint32_t des_xsize)
{
    dev->cfg.ch_base->CH_XSIZE =
        ((des_xsize & 0x0000FFFFUL) << 16U) | (src_xsize & 0x0000FFFFUL);
    dev->cfg.ch_base->CH_XSIZEHI =
        (des_xsize & 0xFFFF0000UL) | ((src_xsize & 0xFFFF0000UL) >> 16U);
}

static __inline
void dma350_ch_set_ysize16(struct dma350_ch_dev_t *dev, uint16_t src_ysize,
                           uint16_t des_ysize)
{
    dev->cfg.ch_base->CH_YSIZE =
        ((des_ysize & 0x0000FFFFUL) << 16U) | (src_ysize & 0x0000FFFFUL);
}

static __inline
void dma350_ch_set_yaddrstride(struct dma350_ch_dev_t *dev,
                               uint16_t src_yaddrstride,
                               uint16_t des_yaddrstride)
{
    dev->cfg.ch_base->CH_YADDRSTRIDE =
        ((des_yaddrstride & 0x0000FFFFUL) << 16U) |
        (src_yaddrstride & 0x0000FFFFUL);
}

static __inline
void dma350_ch_set_transize(struct dma350_ch_dev_t *dev,
                            enum dma350_ch_transize_t transize)
{
    dev->cfg.ch_base->CH_CTRL =
        (dev->cfg.ch_base->CH_CTRL & (~(0x7UL << (0U)))) |
        (transize & (0x7UL << (0U)));
}

static __inline
void dma350_ch_set_chprio(struct dma350_ch_dev_t *dev, uint8_t chprio)
{
    do { dev->cfg.ch_base->CH_CTRL = (dev->cfg.ch_base->CH_CTRL & ~(0xFUL << (4U))) | (((uint32_t)(chprio) << (4U)) & (0xFUL << (4U))); } while (0);

}

static __inline
void dma350_ch_set_xtype(struct dma350_ch_dev_t *dev,
                         enum dma350_ch_xtype_t xtype)
{
    dev->cfg.ch_base->CH_CTRL =
        (dev->cfg.ch_base->CH_CTRL & (~(0x7UL << (9U)))) |
        (xtype & (0x7UL << (9U)));
}

static __inline
void dma350_ch_set_ytype(struct dma350_ch_dev_t *dev,
                         enum dma350_ch_ytype_t ytype)
{
    dev->cfg.ch_base->CH_CTRL =
        (dev->cfg.ch_base->CH_CTRL & (~(0x7UL << (12U)))) |
        (ytype & (0x7UL << (12U)));
}

static __inline
void dma350_ch_set_regreloadtype(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_regreloadtype_t regreloadtype)
{
    dev->cfg.ch_base->CH_CTRL =
        (dev->cfg.ch_base->CH_CTRL & (~(0x7UL << (18U)))) |
        (regreloadtype & (0x7UL << (18U)));
}

static __inline
void dma350_ch_set_donetype(struct dma350_ch_dev_t *dev,
                            enum dma350_ch_donetype_t donetype)
{
    dev->cfg.ch_base->CH_CTRL =
        (dev->cfg.ch_base->CH_CTRL & (~(0x7UL << (21U)))) |
        (donetype & (0x7UL << (21U)));
}

static __inline
void dma350_ch_enable_donepause(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL | (0x1UL << (24U));
}

static __inline
void dma350_ch_disable_donepause(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL & (~(0x1UL << (24U)));
}

static __inline
void dma350_ch_enable_srctrigin(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL | (0x1UL << (25U));
}

static __inline
void dma350_ch_disable_srctrigin(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL & (~(0x1UL << (25U)));
}

static __inline
void dma350_ch_enable_destrigin(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL | (0x1UL << (26U));
}

static __inline
void dma350_ch_disable_destrigin(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL & (~(0x1UL << (26U)));
}

static __inline
void dma350_ch_enable_trigout(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL | (0x1UL << (27U));
}

static __inline
void dma350_ch_disable_trigout(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL & (~(0x1UL << (27U)));
}

static __inline
void dma350_ch_enable_gpo(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL | (0x1UL << (28U));
}

static __inline
void dma350_ch_disable_gpo(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL & (~(0x1UL << (28U)));
}

static __inline
void dma350_ch_enable_stream(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL | (0x1UL << (29U));
}

static __inline
void dma350_ch_disable_stream(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_CTRL =
        dev->cfg.ch_base->CH_CTRL & (~(0x1UL << (29U)));
}

static __inline
void dma350_ch_set_srcmemattr(struct dma350_ch_dev_t *dev, uint8_t memattr,
                              uint8_t shareattr)
{
    dev->cfg.ch_base->CH_SRCTRANSCFG =

        (dev->cfg.ch_base->CH_SRCTRANSCFG &
         ((~(0xFUL << (0U))) |
          (~(0xFUL << (4U))) |
          (~(0x3UL << (8U))))

        ) |
        ((memattr & 0x000000FFUL) << (0U))

        | ((shareattr & 0x00000003UL) << (8U));
}

static __inline
void dma350_ch_set_srcmemattrlo(struct dma350_ch_dev_t *dev, uint8_t memattrlo)
{
    do { dev->cfg.ch_base->CH_SRCTRANSCFG = (dev->cfg.ch_base->CH_SRCTRANSCFG & ~(0xFUL << (0U))) | (((uint32_t)(memattrlo) << (0U)) & (0xFUL << (0U))); } while (0);


}

static __inline
void dma350_ch_set_srcmemattrhi(struct dma350_ch_dev_t *dev, uint8_t memattrhi)
{
    do { dev->cfg.ch_base->CH_SRCTRANSCFG = (dev->cfg.ch_base->CH_SRCTRANSCFG & ~(0xFUL << (4U))) | (((uint32_t)(memattrhi) << (4U)) & (0xFUL << (4U))); } while (0);


}

static __inline
void dma350_ch_set_srcshareattr(struct dma350_ch_dev_t *dev, uint8_t shareattr)
{
    do { dev->cfg.ch_base->CH_SRCTRANSCFG = (dev->cfg.ch_base->CH_SRCTRANSCFG & ~(0x3UL << (8U))) | (((uint32_t)(shareattr) << (8U)) & (0x3UL << (8U))); } while (0);


}

static __inline
void dma350_ch_set_desmemattr(struct dma350_ch_dev_t *dev, uint8_t memattr,
                              uint8_t shareattr)
{
    dev->cfg.ch_base->CH_DESTRANSCFG =

        (dev->cfg.ch_base->CH_DESTRANSCFG &
         ((~(0xFUL << (0U))) |
          (~(0xFUL << (4U))) |
          (~(0x3UL << (8U))))

        ) |
        ((memattr & 0x000000FFUL) << (0U))

        | ((shareattr & 0x00000003UL) << (8U));
}

static __inline
void dma350_ch_set_desmemattrlo(struct dma350_ch_dev_t *dev, uint8_t memattrlo)
{
    do { dev->cfg.ch_base->CH_DESTRANSCFG = (dev->cfg.ch_base->CH_DESTRANSCFG & ~(0xFUL << (0U))) | (((uint32_t)(memattrlo) << (0U)) & (0xFUL << (0U))); } while (0);


}

static __inline
void dma350_ch_set_desmemattrhi(struct dma350_ch_dev_t *dev, uint8_t memattrhi)
{
    do { dev->cfg.ch_base->CH_DESTRANSCFG = (dev->cfg.ch_base->CH_DESTRANSCFG & ~(0xFUL << (4U))) | (((uint32_t)(memattrhi) << (4U)) & (0xFUL << (4U))); } while (0);


}

static __inline
void dma350_ch_set_desshareattr(struct dma350_ch_dev_t *dev, uint8_t shareattr)
{
    do { dev->cfg.ch_base->CH_DESTRANSCFG = (dev->cfg.ch_base->CH_DESTRANSCFG & ~(0x3UL << (8U))) | (((uint32_t)(shareattr) << (8U)) & (0x3UL << (8U))); } while (0);


}

static __inline
void dma350_ch_set_src_trans_secure(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_SRCTRANSCFG = dev->cfg.ch_base->CH_SRCTRANSCFG &
                                       (~(0x1UL << (10U)));
}

static __inline
void dma350_ch_set_src_trans_nonsecure(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_SRCTRANSCFG =
        dev->cfg.ch_base->CH_SRCTRANSCFG | (0x1UL << (10U));
}

static __inline
void dma350_ch_set_des_trans_secure(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_DESTRANSCFG = dev->cfg.ch_base->CH_DESTRANSCFG &
                                       (~(0x1UL << (10U)));
}

static __inline
void dma350_ch_set_des_trans_nonsecure(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_DESTRANSCFG =
        dev->cfg.ch_base->CH_DESTRANSCFG | (0x1UL << (10U));
}

static __inline
void dma350_ch_set_src_trans_privileged(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_SRCTRANSCFG =
        dev->cfg.ch_base->CH_SRCTRANSCFG | (0x1UL << (11U));
}

static __inline
void dma350_ch_set_src_trans_unprivileged(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_SRCTRANSCFG = dev->cfg.ch_base->CH_SRCTRANSCFG &
                                       (~(0x1UL << (11U)));
}

static __inline
void dma350_ch_set_des_trans_privileged(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_DESTRANSCFG =
        dev->cfg.ch_base->CH_DESTRANSCFG | (0x1UL << (11U));
}

static __inline
void dma350_ch_set_des_trans_unprivileged(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_DESTRANSCFG = dev->cfg.ch_base->CH_DESTRANSCFG &
                                       (~(0x1UL << (11U)));
}

static __inline
void dma350_ch_set_srcmaxburstlen(struct dma350_ch_dev_t *dev, uint8_t length)
{
    do { dev->cfg.ch_base->CH_SRCTRANSCFG = (dev->cfg.ch_base->CH_SRCTRANSCFG & ~(0xFUL << (16U))) | (((uint32_t)(length) << (16U)) & (0xFUL << (16U))); } while (0);


}

static __inline
void dma350_ch_set_desmaxburstlen(struct dma350_ch_dev_t *dev, uint8_t length)
{
    do { dev->cfg.ch_base->CH_DESTRANSCFG = (dev->cfg.ch_base->CH_DESTRANSCFG & ~(0xFUL << (16U))) | (((uint32_t)(length) << (16U)) & (0xFUL << (16U))); } while (0);


}

static __inline
void dma350_ch_set_src_xaddr_inc(struct dma350_ch_dev_t *dev,
                                 int16_t src_xaddr_inc)
{
    dev->cfg.ch_base->CH_XADDRINC =
        (dev->cfg.ch_base->CH_XADDRINC & (~(0xFFFFUL << (0U)))) |
        ((uint32_t)src_xaddr_inc & 0x0000FFFFUL);
}

static __inline
void dma350_ch_set_des_xaddr_inc(struct dma350_ch_dev_t *dev,
                                 int16_t des_xaddr_inc)
{
    dev->cfg.ch_base->CH_XADDRINC =
        (dev->cfg.ch_base->CH_XADDRINC & (~(0xFFFFUL << (16U)))) |
        (((uint32_t)des_xaddr_inc & 0x0000FFFFUL) << (16U));
}

static __inline
void dma350_ch_set_xaddr_inc(struct dma350_ch_dev_t *dev, int16_t src_xaddr_inc,
                             int16_t des_xaddr_inc)
{
    dev->cfg.ch_base->CH_XADDRINC =
        (((uint32_t)des_xaddr_inc & 0x0000FFFFUL) << (16U)) |
        ((uint32_t)src_xaddr_inc & 0x0000FFFFUL);
}

static __inline
void dma350_ch_set_fill_value(struct dma350_ch_dev_t *dev, uint32_t fill_value)
{
    dev->cfg.ch_base->CH_FILLVAL = fill_value;
}

static __inline
void dma350_ch_set_srctriginsel(struct dma350_ch_dev_t *dev,
                                uint8_t srctriginsel)
{
    do { dev->cfg.ch_base->CH_SRCTRIGINCFG = (dev->cfg.ch_base->CH_SRCTRIGINCFG & ~(0xFFUL << (0U))) | (((uint32_t)(srctriginsel) << (0U)) & (0xFFUL << (0U))); } while (0);


}

static __inline
void dma350_ch_set_srctrigintype(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_srctrigintype_t type)
{
    dev->cfg.ch_base->CH_SRCTRIGINCFG =
        (dev->cfg.ch_base->CH_SRCTRIGINCFG &
         (~(0x3UL << (8U)))) |
        (type & (0x3UL << (8U)));
}

static __inline
void dma350_ch_set_srctriginmode(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_srctriginmode_t mode)
{
    dev->cfg.ch_base->CH_SRCTRIGINCFG =
        (dev->cfg.ch_base->CH_SRCTRIGINCFG &
         (~(0x3UL << (10U)))) |
        (mode & (0x3UL << (10U)));
}

static __inline
void dma350_ch_set_srctriginblksize(struct dma350_ch_dev_t *dev,
                                    uint8_t blksize)
{
    do { dev->cfg.ch_base->CH_SRCTRIGINCFG = (dev->cfg.ch_base->CH_SRCTRIGINCFG & ~(0xFFUL << (16U))) | (((uint32_t)(blksize) << (16U)) & (0xFFUL << (16U))); } while (0);


}

static __inline
void dma350_ch_set_destriginsel(struct dma350_ch_dev_t *dev,
                                uint8_t destriginsel)
{
    do { dev->cfg.ch_base->CH_DESTRIGINCFG = (dev->cfg.ch_base->CH_DESTRIGINCFG & ~(0xFFUL << (0U))) | (((uint32_t)(destriginsel) << (0U)) & (0xFFUL << (0U))); } while (0);


}

static __inline
void dma350_ch_set_destrigintype(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_destrigintype_t type)
{
    dev->cfg.ch_base->CH_DESTRIGINCFG =
        (dev->cfg.ch_base->CH_DESTRIGINCFG &
         (~(0x3UL << (8U)))) |
        (type & (0x3UL << (8U)));
}

static __inline
void dma350_ch_set_destriginmode(struct dma350_ch_dev_t *dev,
                                 enum dma350_ch_destriginmode_t mode)
{
    dev->cfg.ch_base->CH_DESTRIGINCFG =
        (dev->cfg.ch_base->CH_DESTRIGINCFG &
         (~(0x3UL << (10U)))) |
        (mode & (0x3UL << (10U)));
}

static __inline
void dma350_ch_set_destriginblksize(struct dma350_ch_dev_t *dev,
                                    uint8_t blksize)
{
    do { dev->cfg.ch_base->CH_DESTRIGINCFG = (dev->cfg.ch_base->CH_DESTRIGINCFG & ~(0xFFUL << (16U))) | (((uint32_t)(blksize) << (16U)) & (0xFFUL << (16U))); } while (0);


}

static __inline
void dma350_ch_set_trigoutsel(struct dma350_ch_dev_t *dev, uint8_t trigoutsel)
{
    do { dev->cfg.ch_base->CH_TRIGOUTCFG = (dev->cfg.ch_base->CH_TRIGOUTCFG & ~(0x3FUL << (0U))) | (((uint32_t)(trigoutsel) << (0U)) & (0x3FUL << (0U))); } while (0);


}

static __inline
void dma350_ch_set_trigouttype(struct dma350_ch_dev_t *dev,
                               enum dma350_ch_trigouttype_t type)
{
    dev->cfg.ch_base->CH_TRIGOUTCFG =
        (dev->cfg.ch_base->CH_TRIGOUTCFG &
         (~(0x3UL << (8U)))) |
        (type & (0x3UL << (8U)));
}

static __inline
void dma350_ch_set_gpoen0(struct dma350_ch_dev_t *dev, uint32_t gpoen0)
{
    dev->cfg.ch_base->CH_GPOEN0 = gpoen0;
}

static __inline
void dma350_ch_set_gpoval0(struct dma350_ch_dev_t *dev, uint32_t gpoval0)
{
    dev->cfg.ch_base->CH_GPOVAL0 = gpoval0;
}

static __inline
uint32_t dma350_ch_get_gpoval0(struct dma350_ch_dev_t *dev)
{
    return dev->cfg.ch_base->CH_GPOREAD0;
}

static __inline
void dma350_ch_set_streamtype(struct dma350_ch_dev_t *dev,
                              enum dma350_ch_streamtype_t type)
{
    dev->cfg.ch_base->CH_STREAMINTCFG =
        (dev->cfg.ch_base->CH_STREAMINTCFG &
         (~(0x3UL << (9U)))) |
        (type & (0x3UL << (9U)));
}

static __inline
void dma350_ch_set_linkmemattrlo(struct dma350_ch_dev_t *dev, uint8_t memattrlo)
{
    do { dev->cfg.ch_base->CH_LINKATTR = (dev->cfg.ch_base->CH_LINKATTR & ~(0xFUL << (0U))) | (((uint32_t)(memattrlo) << (0U)) & (0xFUL << (0U))); } while (0);


}

static __inline
void dma350_ch_set_linkmemattrhi(struct dma350_ch_dev_t *dev, uint8_t memattrhi)
{
    do { dev->cfg.ch_base->CH_LINKATTR = (dev->cfg.ch_base->CH_LINKATTR & ~(0xFUL << (4U))) | (((uint32_t)(memattrhi) << (4U)) & (0xFUL << (4U))); } while (0);


}

static __inline
void dma350_ch_set_linkshareattr(struct dma350_ch_dev_t *dev, uint8_t shareattr)
{
    do { dev->cfg.ch_base->CH_LINKATTR = (dev->cfg.ch_base->CH_LINKATTR & ~(0x3UL << (8U))) | (((uint32_t)(shareattr) << (8U)) & (0x3UL << (8U))); } while (0);


}

static __inline
void dma350_ch_enable_linkaddr(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_LINKADDR |= (0x1UL << (0U));
}

static __inline
void dma350_ch_disable_linkaddr(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_LINKADDR &= ~((0x1UL << (0U)));
}

static __inline
void dma350_ch_set_linkaddr32(struct dma350_ch_dev_t *dev, uint32_t linkaddr)
{
    dev->cfg.ch_base->CH_LINKADDR =
        (dev->cfg.ch_base->CH_LINKADDR & (~(0x3FFFFFFFUL << (2U)))) |
        ((uint32_t)linkaddr & (0x3FFFFFFFUL << (2U)));
}

static __inline
void dma350_cmdlink_set_regclear(struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL);
}

static __inline
void dma350_cmdlink_enable_intr(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                enum dma350_ch_intr_t intr)
{
    cmdlink_cfg->header |= (0x1UL << 2);
    cmdlink_cfg->cfg.intren |= intr;
}

static __inline
void dma350_cmdlink_disable_intr(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 enum dma350_ch_intr_t intr)
{
    cmdlink_cfg->header |= (0x1UL << 2);
    cmdlink_cfg->cfg.intren &= (~intr);
}

static __inline
void dma350_cmdlink_set_transize(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 enum dma350_ch_transize_t transize)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl =
        (cmdlink_cfg->cfg.ctrl & (~(0x7UL << (0U)))) |
        (transize & (0x7UL << (0U)));
}

static __inline
void dma350_cmdlink_set_chprio(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                               uint8_t chprio)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl =
        (cmdlink_cfg->cfg.ctrl & (~(0xFUL << (4U)))) |
        (((chprio & 0x000000FFUL) << (4U)) &
         (0xFUL << (4U)));
}

static __inline
void dma350_cmdlink_set_xtype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                              enum dma350_ch_xtype_t xtype)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl = (cmdlink_cfg->cfg.ctrl & (~(0x7UL << (9U)))) |
                            (xtype & (0x7UL << (9U)));
}

static __inline
void dma350_cmdlink_set_ytype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                              enum dma350_ch_ytype_t ytype)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl = (cmdlink_cfg->cfg.ctrl & (~(0x7UL << (12U)))) |
                            (ytype & (0x7UL << (12U)));
}

static __inline
void dma350_cmdlink_set_regreloadtype(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_regreloadtype_t regreloadtype)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl =
        (cmdlink_cfg->cfg.ctrl & (~(0x7UL << (18U)))) |
        (regreloadtype & (0x7UL << (18U)));
}

static __inline
void dma350_cmdlink_set_donetype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 enum dma350_ch_donetype_t donetype)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl =
        (cmdlink_cfg->cfg.ctrl & (~(0x7UL << (21U)))) |
        (donetype & (0x7UL << (21U)));
}

static __inline
void dma350_cmdlink_enable_donepause(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl |= (0x1UL << (24U));
}

static __inline
void dma350_cmdlink_disable_donepause(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl &= (~(0x1UL << (24U)));
}

static __inline
void dma350_cmdlink_enable_srctrigin(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl |= (0x1UL << (25U));
}

static __inline
void dma350_cmdlink_disable_srctrigin(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl &= (~(0x1UL << (25U)));
}

static __inline
void dma350_cmdlink_enable_destrigin(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl |= (0x1UL << (26U));
}

static __inline
void dma350_cmdlink_disable_destrigin(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl &= (~(0x1UL << (26U)));
}

static __inline
void dma350_cmdlink_enable_trigout(struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl |= (0x1UL << (27U));
}

static __inline
void dma350_cmdlink_disable_trigout(struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl &= (~(0x1UL << (27U)));
}

static __inline
void dma350_cmdlink_enable_gpo(struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl |= (0x1UL << (28U));
}

static __inline
void dma350_cmdlink_disable_gpo(struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl &= (~(0x1UL << (28U)));
}

static __inline
void dma350_cmdlink_enable_stream(struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl |= (0x1UL << (29U));
}

static __inline
void dma350_cmdlink_disable_stream(struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 3);
    cmdlink_cfg->cfg.ctrl &= (~(0x1UL << (29U)));
}

static __inline
void dma350_cmdlink_set_srcaddr32(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                  uint32_t src_addr)
{
    cmdlink_cfg->header |= (0x1UL << 4);
    cmdlink_cfg->header &= (~(0x1UL << 5));
    cmdlink_cfg->cfg.srcaddr = src_addr;
}

static __inline
void dma350_cmdlink_set_desaddr32(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                  uint32_t des_addr)
{
    cmdlink_cfg->header |= (0x1UL << 6);
    cmdlink_cfg->header &= (~(0x1UL << 7));
    cmdlink_cfg->cfg.desaddr = des_addr;
}

static __inline
void dma350_cmdlink_set_xsize16(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint16_t src_xsize, uint16_t des_xsize)
{
    cmdlink_cfg->header |= (0x1UL << 8);
    cmdlink_cfg->header &= (~(0x1UL << 9));
    cmdlink_cfg->cfg.xsize = (des_xsize & 0x0000FFFFUL)
                             << (16U);
    cmdlink_cfg->cfg.xsize |= (src_xsize & 0x0000FFFFUL)
                              << (0U);
}

static __inline
void dma350_cmdlink_set_xsize32(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint32_t src_xsize, uint32_t des_xsize)
{
    cmdlink_cfg->header |= (0x1UL << 8);
    cmdlink_cfg->header |= (0x1UL << 9);
    cmdlink_cfg->cfg.xsize = (des_xsize & 0x0000FFFFUL)
                             << (16U);
    cmdlink_cfg->cfg.xsize |= (src_xsize & 0x0000FFFFUL)
                              << (0U);
    cmdlink_cfg->cfg.xsizehi = (des_xsize & 0xFFFF0000UL);
    cmdlink_cfg->cfg.xsizehi |= (src_xsize & 0xFFFF0000UL) >> 16;
}

static __inline
void dma350_cmdlink_set_srcmemattrlo(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrlo)
{
    cmdlink_cfg->header |= (0x1UL << 10);
    cmdlink_cfg->cfg.srctranscfg =
        (cmdlink_cfg->cfg.srctranscfg &
         (~(0xFUL << (0U)))) |
        (((memattrlo & 0x000000FFUL) << (0U)) &
         (0xFUL << (0U)));
}

static __inline
void dma350_cmdlink_set_srcmemattrhi(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrhi)
{
    cmdlink_cfg->header |= (0x1UL << 10);
    cmdlink_cfg->cfg.srctranscfg =
        (cmdlink_cfg->cfg.srctranscfg &
         (~(0xFUL << (4U)))) |
        (((memattrhi & 0x000000FFUL) << (4U)) &
         (0xFUL << (4U)));
}

static __inline
void dma350_cmdlink_set_srcshareattr(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t shareattr)
{
    cmdlink_cfg->header |= (0x1UL << 10);
    cmdlink_cfg->cfg.srctranscfg =
        (cmdlink_cfg->cfg.srctranscfg &
         (~(0x3UL << (8U)))) |
        (((shareattr & 0x000000FFUL) << (8U)) &
         (0x3UL << (8U)));
}

static __inline
void dma350_cmdlink_set_desmemattrlo(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrlo)
{
    cmdlink_cfg->header |= (0x1UL << 11);
    cmdlink_cfg->cfg.destranscfg =
        (cmdlink_cfg->cfg.destranscfg &
         (~(0x3UL << (8U)))) |
        (((memattrlo & 0x000000FFUL) << (0U)) &
         (0xFUL << (0U)));
}

static __inline
void dma350_cmdlink_set_desmemattrhi(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrhi)
{
    cmdlink_cfg->header |= (0x1UL << 11);
    cmdlink_cfg->cfg.destranscfg =
        (cmdlink_cfg->cfg.destranscfg &
         (~(0xFUL << (4U)))) |
        (((memattrhi & 0x000000FFUL) << (4U)) &
         (0xFUL << (4U)));
}

static __inline
void dma350_cmdlink_set_desshareattr(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t shareattr)
{
    cmdlink_cfg->header |= (0x1UL << 11);
    cmdlink_cfg->cfg.destranscfg =
        (cmdlink_cfg->cfg.destranscfg &
         (~(0x3UL << (8U)))) |
        (((shareattr & 0x000000FFUL) << (8U)) &
         (0x3UL << (8U)));
}

static __inline
void dma350_cmdlink_set_src_trans_secure(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 10);
    cmdlink_cfg->cfg.srctranscfg &= (~(0x1UL << (10U)));
}

static __inline
void dma350_cmdlink_set_src_trans_nonsecure(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->cfg.srctranscfg |= (0x1UL << (10U));

    if (cmdlink_cfg->cfg.srctranscfg == 0x000F0400)
    {
        cmdlink_cfg->header &= (~(0x1UL << 10));
    }
}

static __inline
void dma350_cmdlink_set_des_trans_secure(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 11);
    cmdlink_cfg->cfg.destranscfg &= (~(0x1UL << (10U)));
}

static __inline
void dma350_cmdlink_set_des_trans_nonsecure(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->cfg.destranscfg |= (0x1UL << (10U));

    if (cmdlink_cfg->cfg.destranscfg == 0x000F0400)
    {
        cmdlink_cfg->header &= (~(0x1UL << 11));
    }
}

static __inline
void dma350_cmdlink_set_src_trans_privileged(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 10);
    cmdlink_cfg->cfg.srctranscfg |= (0x1UL << (11U));
}

static __inline
void dma350_cmdlink_set_src_trans_unprivileged(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->cfg.srctranscfg &= (~(0x1UL << (11U)));

    if (cmdlink_cfg->cfg.srctranscfg == 0x000F0400)
    {
        cmdlink_cfg->header &= (~(0x1UL << 10));
    }
}

static __inline
void dma350_cmdlink_set_des_trans_privileged(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 11);
    cmdlink_cfg->cfg.destranscfg |= (0x1UL << (11U));
}

static __inline
void dma350_cmdlink_set_des_trans_unprivileged(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->cfg.destranscfg &= (~(0x1UL << (11U)));

    if (cmdlink_cfg->cfg.destranscfg == 0x000F0400)
    {
        cmdlink_cfg->header &= (~(0x1UL << 11));
    }
}

static __inline
void dma350_cmdlink_set_srcmaxburstlen(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t length)
{
    cmdlink_cfg->header |= (0x1UL << 10);
    cmdlink_cfg->cfg.srctranscfg =
        (cmdlink_cfg->cfg.srctranscfg &
         (~(0xFUL << (16U)))) |
        (((length & 0x000000FFUL) << (16U)) &
         (0xFUL << (16U)));
}

static __inline
void dma350_cmdlink_set_desmaxburstlen(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t length)
{
    cmdlink_cfg->header |= (0x1UL << 11);
    cmdlink_cfg->cfg.destranscfg =
        (cmdlink_cfg->cfg.destranscfg &
         (~(0xFUL << (16U)))) |
        (((length & 0x000000FFUL) << (16U)) &
         (0xFUL << (16U)));
}

static __inline
void dma350_cmdlink_set_xaddrinc(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 uint16_t src_xaddrinc, uint16_t des_xaddrinc)
{
    cmdlink_cfg->header |= (0x1UL << 12);
    cmdlink_cfg->cfg.xaddrinc = (des_xaddrinc & 0x0000FFFFUL)
                                << (16U);
    cmdlink_cfg->cfg.xaddrinc |= (src_xaddrinc & 0x0000FFFFUL)
                                 << (0U);
}

static __inline
void dma350_cmdlink_set_yaddrstride(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                    uint16_t src_yaddrstride,
                                    uint16_t des_yaddrstride)
{
    cmdlink_cfg->header |= (0x1UL << 13);
    cmdlink_cfg->cfg.yaddrstride = (des_yaddrstride & 0x0000FFFFUL)
                                   << (16U);
    cmdlink_cfg->cfg.yaddrstride |= (src_yaddrstride & 0x0000FFFFUL)
                                    << (0U);
}

static __inline
void dma350_cmdlink_set_fillval(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint32_t fillval)
{
    cmdlink_cfg->header |= (0x1UL << 14);
    cmdlink_cfg->cfg.fillval = fillval << (0U);
}

static __inline
void dma350_cmdlink_set_ysize16(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint16_t src_ysize, uint16_t des_ysize)
{
    cmdlink_cfg->header |= (0x1UL << 15);
    cmdlink_cfg->cfg.ysize = (des_ysize & 0x0000FFFFUL)
                             << (16U);
    cmdlink_cfg->cfg.ysize |= (src_ysize & 0x0000FFFFUL)
                              << (0U);
}

static __inline
void dma350_cmdlink_set_tmpltsize(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                  uint8_t src_tmpltsize, uint8_t des_tmpltsize)
{
    if (src_tmpltsize <= 0x1FUL &&
            des_tmpltsize <= 0x1FUL)
    {
        cmdlink_cfg->header |= (0x1UL << 16);
        cmdlink_cfg->cfg.tmpltcfg = (des_tmpltsize & 0x000000FFUL)
                                    << (16U);
        cmdlink_cfg->cfg.tmpltcfg |= (src_tmpltsize & 0x000000FFUL)
                                     << (8U);
    }
}

static __inline
void dma350_cmdlink_set_srctmplt(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 uint32_t src_tmplt)
{
    cmdlink_cfg->header |= (0x1UL << 17);
    cmdlink_cfg->cfg.srctmplt = src_tmplt;
}

static __inline
void dma350_cmdlink_set_destmplt(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                 uint32_t des_tmplt)
{
    cmdlink_cfg->header |= (0x1UL << 18);
    cmdlink_cfg->cfg.destmplt = des_tmplt;
}

static __inline
void dma350_cmdlink_set_srctriginsel(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t srctriginsel)
{
    cmdlink_cfg->header |= (0x1UL << 19);
    cmdlink_cfg->cfg.srctrigincfg = (cmdlink_cfg->cfg.srctrigincfg &
                                     (~(0xFFUL << (0U)))) |
                                    (((srctriginsel & 0x000000FFUL)
                                      << (0U)) &
                                     (0xFFUL << (0U)));
}

static __inline
void dma350_cmdlink_set_srctrigintype(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_srctrigintype_t type)
{
    cmdlink_cfg->header |= (0x1UL << 19);
    cmdlink_cfg->cfg.srctrigincfg =
        (cmdlink_cfg->cfg.srctrigincfg &
         (~(0x3UL << (8U)))) |
        (type & (0x3UL << (8U)));
}

static __inline
void dma350_cmdlink_set_srctriginmode(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_srctriginmode_t mode)
{
    cmdlink_cfg->header |= (0x1UL << 19);
    cmdlink_cfg->cfg.srctrigincfg =
        (cmdlink_cfg->cfg.srctrigincfg &
         (~(0x3UL << (10U)))) |
        (mode & (0x3UL << (10U)));
}

static __inline
void dma350_cmdlink_set_srctriginblksize(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t blksize)
{
    cmdlink_cfg->header |= (0x1UL << 19);
    cmdlink_cfg->cfg.srctrigincfg =
        (cmdlink_cfg->cfg.srctrigincfg &
         (~(0xFFUL << (16U)))) |
        (((blksize & 0x000000FFUL)
          << (16U)) &
         (0xFFUL << (16U)));
}

static __inline
void dma350_cmdlink_set_destriginsel(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t destriginsel)
{
    cmdlink_cfg->header |= (0x1UL << 20);
    cmdlink_cfg->cfg.destrigincfg = (cmdlink_cfg->cfg.destrigincfg &
                                     (~(0xFFUL << (0U)))) |
                                    (((destriginsel & 0x000000FFUL)
                                      << (0U)) &
                                     (0xFFUL << (0U)));
}

static __inline
void dma350_cmdlink_set_destrigintype(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_destrigintype_t type)
{
    cmdlink_cfg->header |= (0x1UL << 20);
    cmdlink_cfg->cfg.destrigincfg =
        (cmdlink_cfg->cfg.destrigincfg &
         (~(0x3UL << (8U)))) |
        (type & (0x3UL << (8U)));
}

static __inline
void dma350_cmdlink_set_destriginmode(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
    enum dma350_ch_destriginmode_t mode)
{
    cmdlink_cfg->header |= (0x1UL << 20);
    cmdlink_cfg->cfg.destrigincfg =
        (cmdlink_cfg->cfg.destrigincfg &
         (~(0x3UL << (10U)))) |
        (mode & (0x3UL << (10U)));
}

static __inline
void dma350_cmdlink_set_destriginblksize(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t blksize)
{
    cmdlink_cfg->header |= (0x1UL << 20);
    cmdlink_cfg->cfg.destrigincfg =
        (cmdlink_cfg->cfg.destrigincfg &
         (~(0xFFUL << (16U)))) |
        (((blksize & 0x000000FFUL)
          << (16U)) &
         (0xFFUL << (16U)));
}

static __inline
void dma350_cmdlink_set_trigoutsel(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                   uint8_t trigoutsel)
{
    cmdlink_cfg->header |= (0x1UL << 21);
    cmdlink_cfg->cfg.trigoutcfg =
        (cmdlink_cfg->cfg.trigoutcfg & (~(0x3FUL << (0U)))) |
        (((trigoutsel & 0x000000FFUL) << (0U)) &
         (0x3FUL << (0U)));
}

static __inline
void dma350_cmdlink_set_trigouttype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                    enum dma350_ch_trigouttype_t type)
{
    cmdlink_cfg->header |= (0x1UL << 21);
    cmdlink_cfg->cfg.trigoutcfg =
        (cmdlink_cfg->cfg.trigoutcfg & (~(0x3UL << (8U)))) |
        (type & (0x3UL << (8U)));
}

static __inline
void dma350_cmdlink_set_gpoen0(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                               uint32_t gpoen0)
{
    cmdlink_cfg->header |= (0x1UL << 22);
    cmdlink_cfg->cfg.gpoen0 = gpoen0;
}

static __inline
void dma350_cmdlink_set_gpoval0(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                uint32_t gpoval0)
{
    cmdlink_cfg->header |= (0x1UL << 24);
    cmdlink_cfg->cfg.gpoval0 = gpoval0;
}

static __inline
void dma350_cmdlink_set_streamtype(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                   enum dma350_ch_streamtype_t type)
{
    cmdlink_cfg->header |= (0x1UL << 26);
    cmdlink_cfg->cfg.streamintcfg = (cmdlink_cfg->cfg.streamintcfg &
                                     (~(0x3UL << (9U)))) |
                                    (type & (0x3UL << (9U)));
}

static __inline
void dma350_cmdlink_set_linkmemattrlo(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrlo)
{
    cmdlink_cfg->header |= (0x1UL << 28);
    cmdlink_cfg->cfg.linkattr =
        (cmdlink_cfg->cfg.linkattr & (~(0xFUL << (0U)))) |
        (((memattrlo & 0x000000FFUL) << (0U)) &
         (0xFUL << (0U)));
}

static __inline
void dma350_cmdlink_set_linkmemattrhi(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t memattrhi)
{
    cmdlink_cfg->header |= (0x1UL << 28);
    cmdlink_cfg->cfg.linkattr =
        (cmdlink_cfg->cfg.linkattr & (~(0xFUL << (4U)))) |
        (((memattrhi & 0x000000FFUL) << (4U)) &
         (0xFUL << (4U)));
}

static __inline
void dma350_cmdlink_set_linkshareattr(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint8_t shareattr)
{
    cmdlink_cfg->header |= (0x1UL << 28);
    cmdlink_cfg->cfg.linkattr =
        (cmdlink_cfg->cfg.linkattr & (~(0x3UL << (8U)))) |
        (((shareattr & 0x000000FFUL) << (8U)) &
         (0x3UL << (8U)));
}

static __inline
void dma350_cmdlink_set_cmdrestartcnt(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg, uint16_t cnt)
{
    cmdlink_cfg->header |= (0x1UL << 29);
    cmdlink_cfg->cfg.autocfg = (cnt & 0x0000FFFFUL)
                               << (0U);
}

static __inline
void dma350_cmdlink_enable_cmdrestartinfen(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 29);
    cmdlink_cfg->cfg.autocfg |= (0x1UL << (16U));
}

static __inline
void dma350_cmdlink_disable_cmdrestartinfen(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 29);
    cmdlink_cfg->cfg.autocfg &= (~(0x1UL << (16U)));
}

static __inline
void dma350_cmdlink_enable_linkaddr(struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 30);
    cmdlink_cfg->cfg.linkaddr |= (0x1UL << (0U));
}

static __inline
void dma350_cmdlink_disable_linkaddr(
    struct dma350_cmdlink_gencfg_t *cmdlink_cfg)
{
    cmdlink_cfg->header |= (0x1UL << 30);
    cmdlink_cfg->cfg.linkaddr &= ~((0x1UL << (0U)));
}

static __inline
void dma350_cmdlink_set_linkaddr32(struct dma350_cmdlink_gencfg_t *cmdlink_cfg,
                                   uint32_t linkaddr)
{
    cmdlink_cfg->header |= (0x1UL << 30);
    cmdlink_cfg->header &= ~((0x1UL << 31));
    cmdlink_cfg->cfg.linkaddr =
        (cmdlink_cfg->cfg.linkaddr & (~(0x3FFFFFFFUL << (2U)))) |
        (linkaddr & (0x3FFFFFFFUL << (2U)));
}

static __inline
void dma350_ch_cmd(struct dma350_ch_dev_t *dev, enum dma350_ch_cmd_t cmd)
{
    dev->cfg.ch_base->CH_CMD = cmd;
}

static __inline
enum dma350_ch_cmd_t dma350_ch_get_cmd(struct dma350_ch_dev_t *dev)
{
    return (enum dma350_ch_cmd_t)dev->cfg.ch_base->CH_CMD;
}

static __inline
_Bool dma350_ch_is_busy(struct dma350_ch_dev_t *dev)
{
    return (dev->cfg.ch_base->CH_CMD & DMA350_CH_CMD_ENABLECMD) != 0;
}

static __inline
_Bool dma350_ch_is_ready(struct dma350_ch_dev_t *dev)
{
    return (dev->cfg.ch_base->CH_CMD & DMA350_CH_CMD_ENABLECMD) == 0;
}

static __inline
union dma350_ch_status_t dma350_ch_get_status(struct dma350_ch_dev_t *dev)
{
    return (union dma350_ch_status_t)
    {
        .w = (dev->cfg.ch_base->CH_STATUS)
    };
}

static __inline
void dma350_ch_enable_intr(struct dma350_ch_dev_t *dev,
                           enum dma350_ch_intr_t intr)
{
    dev->cfg.ch_base->CH_INTREN = dev->cfg.ch_base->CH_INTREN | intr;
}

static __inline
void dma350_ch_disable_intr(struct dma350_ch_dev_t *dev,
                            enum dma350_ch_intr_t intr)
{
    dev->cfg.ch_base->CH_INTREN = dev->cfg.ch_base->CH_INTREN & (~intr);
}
# 3699 "../../../../Library/StdDriver/inc\\gdma\\dma350_ch_drv.h"
static __inline
_Bool dma350_ch_is_intr_set(struct dma350_ch_dev_t *dev,
                           enum dma350_ch_intr_t intr)
{
    return (dev->cfg.ch_base->CH_STATUS & intr) != 0;
}

static __inline
_Bool dma350_ch_is_stat_set(struct dma350_ch_dev_t *dev,
                           enum dma350_ch_stat_t stat)
{
    return (dev->cfg.ch_base->CH_STATUS & stat) != 0;
}

static __inline
void dma350_ch_clear_stat(struct dma350_ch_dev_t *dev,
                          enum dma350_ch_stat_t stat)
{
    dev->cfg.ch_base->CH_STATUS = stat;
}

static __inline
void dma350_ch_set_autocfg_restart_cnt(struct dma350_ch_dev_t *dev,
                                       uint16_t cnt)
{
    dev->cfg.ch_base->CH_AUTOCFG = (cnt & (0xFFFFUL << (0U)));
}

static __inline
void dma350_ch_set_autocfg_restart_inf(struct dma350_ch_dev_t *dev)
{
    dev->cfg.ch_base->CH_AUTOCFG = (0x1UL << (16U));
}

static __inline
void dma350_ch_set_tmplt_src_size(struct dma350_ch_dev_t *dev, uint32_t size)
{
    do { dev->cfg.ch_base->CH_TMPLTCFG = (dev->cfg.ch_base->CH_TMPLTCFG & ~(0x1FUL << (8U))) | (((uint32_t)(size) << (8U)) & (0x1FUL << (8U))); } while (0);


}

static __inline
void dma350_ch_set_tmplt_des_size(struct dma350_ch_dev_t *dev, uint32_t size)
{
    do { dev->cfg.ch_base->CH_TMPLTCFG = (dev->cfg.ch_base->CH_TMPLTCFG & ~(0x1FUL << (16U))) | (((uint32_t)(size) << (16U)) & (0x1FUL << (16U))); } while (0);


}

static __inline
void dma350_ch_set_tmplt_src(struct dma350_ch_dev_t *dev, uint32_t tmplt)
{
    dev->cfg.ch_base->CH_SRCTMPLT = tmplt;
}

static __inline
void dma350_ch_set_tmplt_des(struct dma350_ch_dev_t *dev, uint32_t tmplt)
{
    dev->cfg.ch_base->CH_DESTMPLT = tmplt;
}

static __inline
uint32_t dma350_ch_get_tmplt_src(struct dma350_ch_dev_t *dev)
{
    return dev->cfg.ch_base->CH_SRCTMPLT;
}

static __inline
uint32_t dma350_ch_get_tmplt_des(struct dma350_ch_dev_t *dev)
{
    return dev->cfg.ch_base->CH_DESTMPLT;
}
# 12 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h" 2








enum dma350_lib_exec_type_t
{
    DMA350_LIB_EXEC_BLOCKING = 0,

    DMA350_LIB_EXEC_START_ONLY,
    DMA350_LIB_EXEC_IRQ,

};


enum dma350_lib_error_t
{
    DMA350_LIB_ERR_NONE = 0,
    DMA350_LIB_ERR_CMD_ERR,
    DMA350_LIB_ERR_CH_NOT_INIT,
    DMA350_LIB_ERR_CH_NOT_READY,
    DMA350_LIB_ERR_RANGE_NOT_ACCESSIBLE,
    DMA350_LIB_ERR_INVALID_CONFIG_TYPE,
    DMA350_LIB_ERR_CHANNEL_INVALID,
    DMA350_LIB_ERR_DEVICE_INVALID,
    DMA350_LIB_ERR_INVALID_EXEC_TYPE,
    DMA350_LIB_ERR_CFG_ERR,
};


enum dma350_lib_transform_t
{
    DMA350_LIB_TRANSFORM_NONE = 0,
    DMA350_LIB_TRANSFORM_ROTATE_90,
    DMA350_LIB_TRANSFORM_ROTATE_180,
    DMA350_LIB_TRANSFORM_ROTATE_270,
    DMA350_LIB_TRANSFORM_MIRROR_VER,
    DMA350_LIB_TRANSFORM_MIRROR_HOR,
    DMA350_LIB_TRANSFORM_MIRROR_TLBR,
    DMA350_LIB_TRANSFORM_MIRROR_TRBL,
};


struct dma350_remap_range_t
{
    uint32_t begin;
    uint32_t end;
    uint32_t offset;
};


struct dma350_remap_list_t
{
    uint32_t size;
    const struct dma350_remap_range_t *const map;
};


extern const struct dma350_remap_list_t dma350_address_remap;
# 85 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
enum dma350_lib_error_t dma350_lib_set_src(struct dma350_ch_dev_t *dev,
                                           const void *src);
# 98 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
enum dma350_lib_error_t dma350_lib_set_des(struct dma350_ch_dev_t *dev,
                                           void *des);
# 114 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
enum dma350_lib_error_t dma350_lib_set_src_des(struct dma350_ch_dev_t *dev,
                                               const void *src, void *des,
                                               uint32_t src_size,
                                               uint32_t des_size);
# 132 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
enum dma350_lib_error_t dma350_memcpy(struct dma350_ch_dev_t *dev,
                                      const void *src, void *des, uint32_t size,
                                      enum dma350_lib_exec_type_t exec_type);
# 150 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
enum dma350_lib_error_t dma350_memmove(struct dma350_ch_dev_t *dev,
                                       const void *src, void *des, uint32_t size,
                                       enum dma350_lib_exec_type_t exec_type);
# 166 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
enum dma350_lib_error_t dma350_endian_swap(struct dma350_ch_dev_t *dev,
                                           const void *src, void *des,
                                           uint8_t size, uint32_t count);
# 196 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
enum dma350_lib_error_t dma350_draw_from_canvas(struct dma350_ch_dev_t *dev,
                                                const void *src, void *des,
                                                uint32_t src_width, uint16_t src_height,
                                                uint16_t src_line_width,
                                                uint32_t des_width, uint16_t des_height,
                                                uint16_t des_line_width,
                                                enum dma350_ch_transize_t pixelsize,
                                                enum dma350_lib_transform_t transform,
                                                enum dma350_lib_exec_type_t exec_type);
# 229 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
static __inline
enum dma350_lib_error_t dma350_draw_from_bitmap(struct dma350_ch_dev_t *dev,
                                                const void *src, void *des,
                                                uint32_t src_width, uint16_t src_height,
                                                uint32_t des_width, uint16_t des_height,
                                                uint16_t des_line_width,
                                                enum dma350_ch_transize_t pixelsize,
                                                enum dma350_lib_transform_t transform,
                                                enum dma350_lib_exec_type_t exec_type);
# 257 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
static __inline
enum dma350_lib_error_t dma350_2d_copy(
    struct dma350_ch_dev_t *dev,
    const void *src, void *des,
    uint32_t width, uint16_t height,
    enum dma350_ch_transize_t pixelsize,
    enum dma350_lib_transform_t transform,
    enum dma350_lib_exec_type_t exec_type);
# 275 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
static __inline
enum dma350_lib_error_t dma350_clear_done_irq(struct dma350_ch_dev_t *dev);
# 287 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
static __inline
enum dma350_lib_error_t verify_dma350_ch_dev_init(struct dma350_ch_dev_t *dev);
# 299 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
static __inline
enum dma350_lib_error_t verify_dma350_ch_dev_ready(struct dma350_ch_dev_t *dev);
# 312 "../../../../Library/StdDriver/inc\\gdma/dma350_lib.h"
static __inline
enum dma350_lib_error_t dma350_get_status(struct dma350_ch_dev_t *dev,
                                          union dma350_ch_status_t *status);

static __inline
enum dma350_lib_error_t dma350_clear_done_irq(struct dma350_ch_dev_t *dev)
{
    enum dma350_lib_error_t lib_err;
    lib_err = verify_dma350_ch_dev_init(dev);

    if (lib_err != DMA350_LIB_ERR_NONE)
    {
        return lib_err;
    }

    dma350_ch_clear_stat(dev, DMA350_CH_STAT_DONE);

    return DMA350_LIB_ERR_NONE;
}

static __inline
enum dma350_lib_error_t verify_dma350_ch_dev_init(struct dma350_ch_dev_t *dev)
{
    enum dma350_ch_error_t ch_err;

    if (!dev)
    {
        return DMA350_LIB_ERR_DEVICE_INVALID;
    }

    if (!dma350_ch_is_init(dev))
    {
        ch_err = dma350_ch_init(dev);

        if (ch_err != DMA350_CH_ERR_NONE)
        {
            return DMA350_LIB_ERR_CHANNEL_INVALID;
        }
    }

    return DMA350_LIB_ERR_NONE;
}

static __inline
enum dma350_lib_error_t verify_dma350_ch_dev_ready(struct dma350_ch_dev_t *dev)
{
    enum dma350_lib_error_t lib_err;
    lib_err = verify_dma350_ch_dev_init(dev);

    if (lib_err != DMA350_LIB_ERR_NONE)
    {
        return lib_err;
    }

    if (!dma350_ch_is_ready(dev))
    {
        return DMA350_LIB_ERR_CH_NOT_READY;
    }

    return DMA350_LIB_ERR_NONE;
}

static __inline
enum dma350_lib_error_t dma350_get_status(struct dma350_ch_dev_t *dev,
                                          union dma350_ch_status_t *status)
{
    enum dma350_lib_error_t lib_err;
    lib_err = verify_dma350_ch_dev_init(dev);

    if (lib_err != DMA350_LIB_ERR_NONE)
    {
        return lib_err;
    }

    *status = dma350_ch_get_status(dev);
    return DMA350_LIB_ERR_NONE;
}

static __inline
enum dma350_lib_error_t dma350_draw_from_bitmap(struct dma350_ch_dev_t *dev,
                                                const void *src, void *des,
                                                uint32_t src_width, uint16_t src_height,
                                                uint32_t des_width, uint16_t des_height,
                                                uint16_t des_line_width,
                                                enum dma350_ch_transize_t pixelsize,
                                                enum dma350_lib_transform_t transform,
                                                enum dma350_lib_exec_type_t exec_type)
{
    return dma350_draw_from_canvas(dev, src, des,
                                   src_width, src_height, (uint16_t)src_width,
                                   des_width, des_height, des_line_width,
                                   pixelsize, transform, exec_type);
}

static __inline
enum dma350_lib_error_t dma350_2d_copy(
    struct dma350_ch_dev_t *dev,
    const void *src, void *des,
    uint32_t width, uint16_t height,
    enum dma350_ch_transize_t pixelsize,
    enum dma350_lib_transform_t transform,
    enum dma350_lib_exec_type_t exec_type)
{
    return dma350_draw_from_canvas(dev, src, des,
                                   width, height, (uint16_t)width,
                                   width, height, (uint16_t)width,
                                   pixelsize, transform, exec_type);
}
# 1701 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h" 1
# 25 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
# 1 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stddef.h" 1 3
# 38 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stddef.h" 3
  typedef signed int ptrdiff_t;
# 95 "C:\\Keil_v5\\ARM\\ARMCLANG\\bin\\..\\include\\stddef.h" 3
  typedef long double max_align_t;
# 26 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h" 2
# 56 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t
{
    DMA350_ERR_NONE = 0,
    DMA350_ERR_IIDR_MISMATCH,

    DMA350_ERR_AIDR_MISMATCH,

    DMA350_ERR_NOT_INIT,
    DMA350_ERR_CANNOT_SET_NOW,
    DMA350_ERR_INVALID_PARAM,
};


union dma350_dmainfo_buildcfg0_t
{
    struct
    {
        uint32_t FRAMETYPE: 3;
        uint32_t RESERVED0: 1;
        uint32_t NUM_CHANNELS: 6;
        uint32_t ADDR_WIDTH: 6;
        uint32_t DATA_WIDTH: 3;
        uint32_t RESERVED1: 1;
        uint32_t CHID_WIDTH: 5;
        uint32_t RESERVED2: 7;
    } b;
    uint32_t w;
};


union dma350_dmainfo_buildcfg1_t
{
    struct
    {
        uint32_t NUM_TRIGGER_IN: 9;
        uint32_t NUM_TRIGGER_OUT: 7;
        uint32_t HAS_TRIGSEL: 1;
        uint32_t RESERVED0: 7;
        uint32_t RESERVED1: 1;
        uint32_t RESERVED2: 7;
    } b;
    uint32_t w;
};


struct dma350_dev_cfg_t
{
    DMASECCFG_TypeDef *dma_sec_cfg;
    DMASECCTRL_TypeDef *dma_sec_ctrl;
    DMANSECCTRL_TypeDef *dma_nsec_ctrl;
    DMAINFO_TypeDef *dma_info;
};


struct dma350_dev_data_t
{
    uint32_t state;

};


struct dma350_dev_t
{
    const struct dma350_dev_cfg_t *const cfg;
    struct dma350_dev_data_t *const data;
};
# 132 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t dma350_init(struct dma350_dev_t *dev);
# 143 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
static __inline
uint8_t dma350_is_init(const struct dma350_dev_t *dev);
# 155 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
static __inline
uint8_t dma350_get_num_ch(const struct dma350_dev_t *dev);
# 167 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
void dma350_enable_secaccvio_irq(struct dma350_dev_t *dev);
# 178 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
void dma350_disable_secaccvio_irq(struct dma350_dev_t *dev);
# 189 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
void dma350_set_secaccvio_rsp_buserr(struct dma350_dev_t *dev);
# 200 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
void dma350_set_secaccvio_rsp_razwi(struct dma350_dev_t *dev);
# 213 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
void dma350_lock_security_cfg(struct dma350_dev_t *dev);
# 226 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
uint8_t dma350_get_secaccvio_irq(struct dma350_dev_t *dev);
# 237 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
uint8_t dma350_get_secaccvio_status(struct dma350_dev_t *dev);
# 248 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
void dma350_clear_secaccvio_status(struct dma350_dev_t *dev);
# 261 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t dma350_set_ch_secure(struct dma350_dev_t *dev,
                                         uint8_t channel);
# 275 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t dma350_set_ch_nonsecure(struct dma350_dev_t *dev,
                                            uint8_t channel);
# 296 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t dma350_set_ch_privileged(struct dma350_dev_t *dev,
                                             uint8_t channel);
# 310 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t dma350_set_ch_unprivileged(struct dma350_dev_t *dev,
                                               uint8_t channel);
# 324 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t dma350_set_trigin_secure(struct dma350_dev_t *dev,
                                             uint8_t trigger);
# 338 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t dma350_set_trigin_nonsecure(struct dma350_dev_t *dev,
                                                uint8_t trigger);
# 352 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t dma350_set_trigout_secure(struct dma350_dev_t *dev,
                                              uint8_t trigger);
# 366 "../../../../Library/StdDriver/inc\\gdma/dma350_drv.h"
enum dma350_error_t dma350_set_trigout_nonsecure(struct dma350_dev_t *dev,
                                                 uint8_t trigger);

static __inline
uint8_t dma350_get_num_ch(const struct dma350_dev_t *dev)
{
    return (union dma350_dmainfo_buildcfg0_t)
    {
        .w = dev->cfg->dma_info->DMA_BUILDCFG0
    } .b.NUM_CHANNELS;
}

static __inline
uint8_t dma350_is_init(const struct dma350_dev_t *dev)
{
    return dev->data->state == (1UL << 0);
}
# 1702 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\gpio.h" 1
# 1397 "../../../../Library/StdDriver/inc\\gpio.h"
void GPIO_SetMode(GPIO_T *port, uint32_t u32PinMask, uint32_t u32Mode);
void GPIO_EnableInt(GPIO_T *port, uint32_t u32Pin, uint32_t u32IntAttribs);
void GPIO_DisableInt(GPIO_T *port, uint32_t u32Pin);
void GPIO_SetSlewCtl(GPIO_T *port, uint32_t u32PinMask, uint32_t u32Mode);
void GPIO_SetPullCtl(GPIO_T *port, uint32_t u32PinMask, uint32_t u32Mode);
void GPIO_EnableEINT(uint32_t u32EINTn, uint32_t u32IntAttribs);
void GPIO_DisableEINT(uint32_t u32EINTn);
# 1703 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\hsotg.h" 1
# 1704 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\hsusbd.h" 1
# 114 "../../../../Library/StdDriver/inc\\hsusbd.h"
typedef struct HSUSBD_CMD_STRUCT
{
    uint8_t bmRequestType;
    uint8_t bRequest;
    uint16_t wValue;
    uint16_t wIndex;
    uint16_t wLength;

} S_HSUSBD_CMD_T;




typedef struct s_hsusbd_info
{
    uint8_t *gu8DevDesc;
    uint8_t *gu8ConfigDesc;
    uint8_t **gu8StringDesc;
    uint8_t *gu8QualDesc;
    uint8_t *gu8FullConfigDesc;
    uint8_t *gu8HSOtherConfigDesc;
    uint8_t *gu8FSOtherConfigDesc;
    uint8_t *gu8BosDesc;
    uint8_t **gu8HidReportDesc;
    uint32_t *gu32HidReportSize;
    uint32_t *gu32ConfigHidDescIdx;

} S_HSUSBD_INFO_T;





extern uint32_t g_u32HsEpStallLock;
extern uint8_t volatile g_hsusbd_Configured;
extern uint8_t g_hsusbd_ShortPacket;
extern uint8_t g_hsusbd_CtrlZero;
extern uint8_t g_hsusbd_UsbAddr;
extern uint32_t volatile g_hsusbd_DmaDone;
extern uint32_t g_hsusbd_CtrlInSize;
extern S_HSUSBD_INFO_T gsHSInfo;
extern S_HSUSBD_CMD_T gUsbCmd;
extern volatile uint8_t g_hsusbd_RemoteWakeupEn;
# 202 "../../../../Library/StdDriver/inc\\hsusbd.h"
static __inline void HSUSBD_MemCopy(uint8_t u8Dst[], uint8_t u8Src[], uint32_t u32Size)
{
    uint32_t i = 0ul;

    while (u32Size--)
    {
        u8Dst[i] = u8Src[i];
        i++;
    }
}






static __inline void HSUSBD_ResetDMA(void)
{
    ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->DMACNT = 0ul;
    ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->DMACTL = 0x80ul;
    ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->DMACTL = 0x00ul;
}







static __inline void HSUSBD_SetEpBufAddr(uint32_t u32Ep, uint32_t u32Base, uint32_t u32Len)
{
    if (u32Ep == 0xfful)
    {
        ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->CEPBUFST = u32Base;
        ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->CEPBUFEND = u32Base + u32Len - 1ul;
    }
    else
    {
        ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPBUFST = u32Base;
        ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPBUFEND = u32Base + u32Len - 1ul;
    }
}
# 253 "../../../../Library/StdDriver/inc\\hsusbd.h"
static __inline void HSUSBD_ConfigEp(uint32_t u32Ep, uint32_t u32EpNum, uint32_t u32EpType, uint32_t u32EpDir)
{
    if (u32EpType == ((uint32_t)0x00000002ul))
    {
        ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPRSPCTL = (((uint32_t)0x00000001ul) | ((uint32_t)0x00000000ul));
    }
    else if (u32EpType == ((uint32_t)0x00000004ul))
    {
        ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPRSPCTL = (((uint32_t)0x00000001ul) | ((uint32_t)0x00000002ul));
    }
    else if (u32EpType == ((uint32_t)0x00000006ul))
    {
        ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPRSPCTL = (((uint32_t)0x00000001ul) | ((uint32_t)0x00000004ul));
    }

    ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPCFG = (u32EpType | u32EpDir | ((uint32_t)0x00000001ul) | (u32EpNum << 4));
}







static __inline void HSUSBD_SetEpStall(uint32_t u32Ep)
{
    if (u32Ep == 0xfful)
    {
        (((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->CEPCTL = (((uint32_t)0x00000002ul)));
    }
    else
    {
        ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPRSPCTL = (((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPRSPCTL & 0xf7ul) | ((uint32_t)0x00000010ul);
    }
}
# 297 "../../../../Library/StdDriver/inc\\hsusbd.h"
static __inline void HSUSBD_SetStall(uint32_t u32EpNum)
{
    uint32_t i;

    if (u32EpNum == 0ul)
    {
        (((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->CEPCTL = (((uint32_t)0x00000002ul)));
    }
    else
    {
        for (i = 0ul; i < 18ul; i++)
        {
            if (((((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[i].EPCFG & 0xf0ul) >> 4) == u32EpNum)
            {
                ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[i].EPRSPCTL = (((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[i].EPRSPCTL & 0xf7ul) | ((uint32_t)0x00000010ul);
            }
        }
    }
}







static __inline void HSUSBD_ClearEpStall(uint32_t u32Ep)
{
    ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPRSPCTL = ((uint32_t)0x00000008ul);
}
# 336 "../../../../Library/StdDriver/inc\\hsusbd.h"
static __inline void HSUSBD_ClearStall(uint32_t u32EpNum)
{
    uint32_t i;

    for (i = 0ul; i < 18ul; i++)
    {
        if (((((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[i].EPCFG & 0xf0ul) >> 4) == u32EpNum)
        {
            ((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[i].EPRSPCTL = ((uint32_t)0x00000008ul);
        }
    }
}
# 356 "../../../../Library/StdDriver/inc\\hsusbd.h"
static __inline uint32_t HSUSBD_GetEpStall(uint32_t u32Ep)
{
    return (((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[u32Ep].EPRSPCTL & ((uint32_t)0x00000010ul));
}
# 370 "../../../../Library/StdDriver/inc\\hsusbd.h"
static __inline uint32_t HSUSBD_GetStall(uint32_t u32EpNum)
{
    uint32_t i;
    uint32_t val = 0ul;

    for (i = 0ul; i < 18ul; i++)
    {
        if (((((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[i].EPCFG & 0xf0ul) >> 4) == u32EpNum)
        {
            val = (((HSUSBD_T *) ((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x05000UL))->EP[i].EPRSPCTL & ((uint32_t)0x00000010ul));
            break;
        }
    }

    return val;
}


typedef void (*HSUSBD_VENDOR_REQ)(void);
typedef void (*HSUSBD_CLASS_REQ)(void);
typedef void (*HSUSBD_SET_INTERFACE_REQ)(uint32_t u32AltInterface);

int32_t HSUSBD_Open(S_HSUSBD_INFO_T *param, HSUSBD_CLASS_REQ pfnClassReq, HSUSBD_SET_INTERFACE_REQ pfnSetInterface);
void HSUSBD_Start(void);
void HSUSBD_ProcessSetupPacket(void);
void HSUSBD_StandardRequest(void);
void HSUSBD_UpdateDeviceState(void);
void HSUSBD_PrepareCtrlIn(uint8_t pu8Buf[], uint32_t u32Size);
void HSUSBD_CtrlIn(void);
int32_t HSUSBD_CtrlOut(uint8_t pu8Buf[], uint32_t u32Size);
void HSUSBD_SwReset(void);
void HSUSBD_SetVendorRequest(HSUSBD_VENDOR_REQ pfnVendorReq);
# 1705 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\i2c.h" 1
# 74 "../../../../Library/StdDriver/inc\\i2c.h"
extern int32_t g_I2C_i32ErrCode;
# 433 "../../../../Library/StdDriver/inc\\i2c.h"
static __inline void I2C_STOP(I2C_T *i2c);
# 443 "../../../../Library/StdDriver/inc\\i2c.h"
static __inline void I2C_STOP(I2C_T *i2c)
{
    uint32_t u32TimeOutCount = SystemCoreClock;

    (i2c)->CTL0 |= ((0x1ul << (3)) | (0x1ul << (4)));

    while (i2c->CTL0 & (0x1ul << (4)))
    {
        if (--u32TimeOutCount == 0) break;
    }
}

void I2C_ClearTimeoutFlag(I2C_T *i2c);
void I2C_Close(I2C_T *i2c);
void I2C_Trigger(I2C_T *i2c, uint8_t u8Start, uint8_t u8Stop, uint8_t u8Si, uint8_t u8Ack);
void I2C_DisableInt(I2C_T *i2c);
void I2C_EnableInt(I2C_T *i2c);
uint32_t I2C_GetBusClockFreq(I2C_T *i2c);
uint32_t I2C_GetIntFlag(I2C_T *i2c);
uint32_t I2C_GetStatus(I2C_T *i2c);
uint32_t I2C_Open(I2C_T *i2c, uint32_t u32BusClock);
uint8_t I2C_GetData(I2C_T *i2c);
void I2C_SetSlaveAddr(I2C_T *i2c, uint8_t u8SlaveNo, uint16_t u16SlaveAddr, uint8_t u8GCMode);
void I2C_SetSlaveAddrMask(I2C_T *i2c, uint8_t u8SlaveNo, uint16_t u16SlaveAddrMask);
uint32_t I2C_SetBusClockFreq(I2C_T *i2c, uint32_t u32BusClock);
void I2C_EnableTimeout(I2C_T *i2c, uint8_t u8LongTimeout);
void I2C_DisableTimeout(I2C_T *i2c);
void I2C_EnableWakeup(I2C_T *i2c);
void I2C_DisableWakeup(I2C_T *i2c);
void I2C_SetData(I2C_T *i2c, uint8_t u8Data);
void I2C_EnableTwoBufferMode(I2C_T *i2c, uint32_t u32BitCount);
void I2C_DisableTwoBufferMode(I2C_T *i2c);
uint8_t I2C_WriteByte(I2C_T *i2c, uint8_t u8SlaveAddr, uint8_t data);
uint32_t I2C_WriteMultiBytes(I2C_T *i2c, uint8_t u8SlaveAddr, uint8_t data[], uint32_t u32wLen);
uint8_t I2C_WriteByteOneReg(I2C_T *i2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr, uint8_t data);
uint32_t I2C_WriteMultiBytesOneReg(I2C_T *i2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr, uint8_t data[], uint32_t u32wLen);
uint8_t I2C_WriteByteTwoRegs(I2C_T *i2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr, uint8_t data);
uint32_t I2C_WriteMultiBytesTwoRegs(I2C_T *i2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr, uint8_t data[], uint32_t u32wLen);
uint8_t I2C_ReadByte(I2C_T *i2c, uint8_t u8SlaveAddr);
uint32_t I2C_ReadMultiBytes(I2C_T *i2c, uint8_t u8SlaveAddr, uint8_t rdata[], uint32_t u32rLen);
uint8_t I2C_ReadByteOneReg(I2C_T *i2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr);
uint32_t I2C_ReadMultiBytesOneReg(I2C_T *i2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr, uint8_t rdata[], uint32_t u32rLen);
uint8_t I2C_ReadByteTwoRegs(I2C_T *i2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr);
uint32_t I2C_ReadMultiBytesTwoRegs(I2C_T *i2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr, uint8_t rdata[], uint32_t u32rLen);
uint32_t I2C_SMBusGetStatus(I2C_T *i2c);
void I2C_SMBusClearInterruptFlag(I2C_T *i2c, uint8_t u8SMBusIntFlag);
void I2C_SMBusSetPacketByteCount(I2C_T *i2c, uint32_t u32PktSize);
void I2C_SMBusOpen(I2C_T *i2c, uint8_t u8HostDevice);
void I2C_SMBusClose(I2C_T *i2c);
void I2C_SMBusPECTxEnable(I2C_T *i2c, uint8_t u8PECTxEn);
uint8_t I2C_SMBusGetPECValue(I2C_T *i2c);
void I2C_SMBusIdleTimeout(I2C_T *i2c, uint32_t us, uint32_t u32Hclk);
void I2C_SMBusTimeout(I2C_T *i2c, uint32_t ms, uint32_t u32Pclk);
void I2C_SMBusClockLoTimeout(I2C_T *i2c, uint32_t ms, uint32_t u32Pclk);
# 1706 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\i2s.h" 1
# 121 "../../../../Library/StdDriver/inc\\i2s.h"
static __inline void I2S_ENABLE_TX_ZCD(I2S_T *i2s, uint32_t u32ChMask)
{
    if ((u32ChMask > 0U) && (u32ChMask < 9U))
    {
        i2s->CTL1 |= ((uint32_t)1U << (u32ChMask - 1U));
    }
}







static __inline void I2S_DISABLE_TX_ZCD(I2S_T *i2s, uint32_t u32ChMask)
{
    if ((u32ChMask > 0U) && (u32ChMask < 9U))
    {
        i2s->CTL1 &= ~((uint32_t)1U << (u32ChMask - 1U));
    }
}
# 233 "../../../../Library/StdDriver/inc\\i2s.h"
static __inline void I2S_SET_MONO_RX_CHANNEL(I2S_T *i2s, uint32_t u32Ch)
{
    u32Ch == (0x1ul << (23)) ?
    (i2s->CTL0 |= (0x1ul << (23))) :
    (i2s->CTL0 &= ~(0x1ul << (23)));
}
# 314 "../../../../Library/StdDriver/inc\\i2s.h"
uint32_t I2S_GetSourceClockFreq(I2S_T *i2s);
uint32_t I2S_Open(I2S_T *i2s, uint32_t u32MasterSlave, uint32_t u32SampleRate, uint32_t u32WordWidth, uint32_t u32MonoData, uint32_t u32DataFormat);
void I2S_Close(I2S_T *i2s);
void I2S_EnableInt(I2S_T *i2s, uint32_t u32Mask);
void I2S_DisableInt(I2S_T *i2s, uint32_t u32Mask);
uint32_t I2S_EnableMCLK(I2S_T *i2s, uint32_t u32BusClock);
void I2S_DisableMCLK(I2S_T *i2s);
void I2S_SetFIFO(I2S_T *i2s, uint32_t u32TxThreshold, uint32_t u32RxThreshold);
void I2S_ConfigureTDM(I2S_T *i2s, uint32_t u32ChannelWidth, uint32_t u32ChannelNum, uint32_t u32SyncWidth);
# 1707 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\i3c.h" 1
# 756 "../../../../Library/StdDriver/inc\\i3c.h"
static __inline int32_t I3C_Enable(I3C_T *i3c);
static __inline int32_t I3C_Disable(I3C_T *i3c);
static __inline void I3C_EnableDMA(I3C_T *i3c);
static __inline void I3C_DisableDMA(I3C_T *i3c);
# 771 "../../../../Library/StdDriver/inc\\i3c.h"
static __inline int32_t I3C_Enable(I3C_T *i3c)
{
    volatile uint32_t u32Timeout;

    i3c->DEVCTL |= (1UL << (31U));
    u32Timeout = (SystemCoreClock / 1000);

    while (((i3c->DEVCTL & (1UL << (31U))) != (1UL << (31U))) && (--u32Timeout)) {}

    if (u32Timeout == 0)
    {
        return (-1L);
    }

    return (0L);
}
# 798 "../../../../Library/StdDriver/inc\\i3c.h"
static __inline int32_t I3C_Disable(I3C_T *i3c)
{
    volatile uint32_t u32Timeout;

    i3c->DEVCTL &= ~(1UL << (31U));
    u32Timeout = (SystemCoreClock / 1000);

    while (((i3c->DEVCTL & (1UL << (31U))) == (1UL << (31U))) && (--u32Timeout)) {}

    if (u32Timeout == 0)
    {
        return (-1L);
    }

    return (0L);
}
# 822 "../../../../Library/StdDriver/inc\\i3c.h"
static __inline void I3C_EnableDMA(I3C_T *i3c)
{
    i3c->DEVCTL &= ~(1UL << (28U));
    i3c->DEVCTL |= (1UL << (28U));
}
# 835 "../../../../Library/StdDriver/inc\\i3c.h"
static __inline void I3C_DisableDMA(I3C_T *i3c)
{
    i3c->DEVCTL &= ~(1UL << (28U));
}


void I3C_Open(I3C_T *i3c, uint32_t u32MasterSlave, uint8_t u8StaticAddr, uint32_t u32ModeSel);
int32_t I3C_ResetAndResume(I3C_T *i3c, uint32_t u32ResetMask, uint32_t u32EnableResume);
int32_t I3C_ParseRespQueue(I3C_T *i3c, uint32_t *pu32RespQ);
int32_t I3C_SetCmdQueueAndData(I3C_T *i3c, uint8_t u8TID, uint32_t *pu32TxBuf, uint16_t u16WriteBytes);
int32_t I3C_SendIBIRequest(I3C_T *i3c, uint8_t u8MandatoryData, uint32_t u32PayloadData, uint8_t u8PayloadLen);
int32_t I3C_EnableHJRequest(I3C_T *i3c, uint32_t u32ModeSel);
int32_t I3C_DisableHJRequest(I3C_T *i3c);
int32_t I3C_RespErrorRecovery(I3C_T *i3c, uint32_t u32RespStatus);
int32_t I3C_SetDeviceAddr(I3C_T *i3c, uint8_t u8DevIndex, uint8_t u8DevType, uint8_t u8DAddr, int8_t u8SAddr);
int32_t I3C_Write(I3C_T *i3c, uint8_t u8DevIndex, uint32_t u32Speed, uint32_t *pu32TxBuf, uint16_t u16WriteBytes);
int32_t I3C_Read(I3C_T *i3c, uint8_t u8DevIndex, uint32_t u32Speed, uint32_t *pu32RxBuf, uint16_t u16ReadBytes);
int32_t I3C_BroadcastRSTDAA(I3C_T *i3c);
int32_t I3C_BroadcastENTDAA(I3C_T *i3c, uint8_t u8DevCount);
int32_t I3C_UnicastSETDASA(I3C_T *i3c, uint8_t u8DevIndex);
# 1708 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\ks.h" 1
# 40 "../../../../Library/StdDriver/inc\\ks.h"
typedef enum KSMEM
{
    KS_SRAM = 0,
    KS_FLASH = 1,
    KS_OTP = 2
} KS_MEM_Type;
# 156 "../../../../Library/StdDriver/inc\\ks.h"
extern int32_t g_KS_i32ErrCode;





int32_t KS_Open(void);
int32_t KS_Read(KS_MEM_Type eMemType, int32_t i32KeyIdx, uint32_t au32Key[], uint32_t u32WordCnt);
int32_t KS_Write(KS_MEM_Type eMemType, uint32_t u32Meta, uint32_t au32Key[]);
int32_t KS_WriteOTP(int32_t i32KeyIdx, uint32_t u32Meta, uint32_t au32Key[]);
int32_t KS_EraseKey(int32_t i32KeyIdx);
int32_t KS_EraseOTPKey(int32_t i32KeyIdx);
int32_t KS_LockOTPKey(int32_t i32KeyIdx);
int32_t KS_EraseAll(KS_MEM_Type eMemType);
int32_t KS_RevokeKey(KS_MEM_Type eMemType, int32_t i32KeyIdx);
uint32_t KS_GetRemainSize(KS_MEM_Type eMemType);
int32_t KS_ToggleSRAM(void);
uint32_t KS_GetKeyWordCnt(uint32_t u32Meta);
uint32_t KS_GetRemainKeyCount(KS_MEM_Type eMemType);
# 1709 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\kdf.h" 1
# 66 "../../../../Library/StdDriver/inc\\kdf.h"
typedef enum
{
    eKDF_ERRCODE_SUCCESS = 0,
    eKDF_ERRCODE_FAIL = -1,
    eKDF_ERRCODE_TIMEOUT = -2,
    eKDF_ERRCODE_BUSY = -3,
    eKDF_ERRCODE_INVALID_PARAM = -4,
} E_KDF_ERRCODE;

typedef enum
{
    eKDF_MODE_HKDF = (0UL << (4)),
    eKDF_MODE_KBKDF = (1UL << (4)),
} E_KDF_MODE;



extern int32_t g_KDF_i32ErrCode;





void KDF_SetKeyInput(const uint8_t pu8KeyInput[], uint32_t u32ByteCnt);
void KDF_SetSalt(const uint8_t pu8Salt [], uint32_t u32ByteCnt);
void KDF_SetLabel(const uint8_t pu8Label [], uint32_t u32ByteCnt);
void KDF_SetContext(const uint8_t pu8Context [], uint32_t u32ByteCnt);
int32_t KDF_GetKeyBitSize(uint32_t u32KeySizeOpt);
int32_t KDF_DeriveKey(E_KDF_MODE eMode, uint32_t u32DeriveKeyParam, uint32_t u32KeyBitSize, uint32_t *pu32KeyOut);
int32_t KDF_DeriveKeyToKS(KS_MEM_Type eMemType, E_KDF_MODE eMode, uint32_t u32DeriveKeyParam, uint32_t u32KeySizeSel, uint32_t u32KeyMeta);
# 1710 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\crypto.h" 1
# 122 "../../../../Library/StdDriver/inc\\crypto.h"
typedef enum
{

    CURVE_P_192 = 0x01,
    CURVE_P_224 = 0x02,
    CURVE_P_256 = 0x03,
    CURVE_P_384 = 0x04,
    CURVE_P_521 = 0x05,
    CURVE_K_163 = 0x11,
    CURVE_K_233 = 0x12,
    CURVE_K_283 = 0x13,
    CURVE_K_409 = 0x14,
    CURVE_K_571 = 0x15,
    CURVE_B_163 = 0x21,
    CURVE_B_233 = 0x22,
    CURVE_B_283 = 0x23,
    CURVE_B_409 = 0x24,
    CURVE_B_571 = 0x25,
    CURVE_KO_192 = 0x31,
    CURVE_KO_224 = 0x32,
    CURVE_KO_256 = 0x33,
    CURVE_BP_256 = 0x41,
    CURVE_BP_384 = 0x42,
    CURVE_BP_512 = 0x43,
    CURVE_SM2_256 = 0x50,
    CURVE_25519 = 0x51,
    CURVE_Ed448 = 0x52,
    CURVE_UNDEF,
}
E_ECC_CURVE;

typedef struct e_curve_t
{
    E_ECC_CURVE curve_id;
    int32_t Echar;
    char Ea[144];
    char Eb[144];
    char Px[144];
    char Py[144];
    int32_t Epl;
    char Pp[176];
    int32_t Eol;
    char Eorder[176];
    int32_t key_len;
    int32_t irreducible_k1;
    int32_t irreducible_k2;
    int32_t irreducible_k3;
    int32_t GF;
} ECC_CURVE;

enum
{
    CURVE_GF_P,
    CURVE_GF_2M,
};




typedef struct
{
# 197 "../../../../Library/StdDriver/inc\\crypto.h"
    uint32_t au32RsaOutput[128];
    uint32_t au32RsaN[128];
    uint32_t au32RsaM[128];
    uint32_t au32RsaE[128];
} RSA_BUF_NORMAL_T;


typedef struct
{
    uint32_t au32RsaOutput[128];
    uint32_t au32RsaN[128];
    uint32_t au32RsaM[128];
    uint32_t au32RsaE[128];
    uint32_t au32RsaP[128];
    uint32_t au32RsaQ[128];
    uint32_t au32RsaTmpCp[128];
    uint32_t au32RsaTmpCq[128];
    uint32_t au32RsaTmpDp[128];
    uint32_t au32RsaTmpDq[128];
    uint32_t au32RsaTmpRp[128];
    uint32_t au32RsaTmpRq[128];
} RSA_BUF_CRT_T;


typedef struct
{
    uint32_t au32RsaOutput[128];
    uint32_t au32RsaN[128];
    uint32_t au32RsaM[128];
    uint32_t au32RsaE[128];
    uint32_t au32RsaP[128];
    uint32_t au32RsaQ[128];
    uint32_t au32RsaTmpBlindKey[128];
} RSA_BUF_SCAP_T;


typedef struct
{
    uint32_t au32RsaOutput[128];
    uint32_t au32RsaN[128];
    uint32_t au32RsaM[128];
    uint32_t au32RsaE[128];
    uint32_t au32RsaP[128];
    uint32_t au32RsaQ[128];
    uint32_t au32RsaTmpCp[128];
    uint32_t au32RsaTmpCq[128];
    uint32_t au32RsaTmpDp[128];
    uint32_t au32RsaTmpDq[128];
    uint32_t au32RsaTmpRp[128];
    uint32_t au32RsaTmpRq[128];
    uint32_t au32RsaTmpBlindKey[128];
} RSA_BUF_CRT_SCAP_T;


typedef struct
{
    uint32_t au32RsaOutput[128];
    uint32_t au32RsaN[128];
    uint32_t au32RsaM[128];
} RSA_BUF_KS_T;
# 475 "../../../../Library/StdDriver/inc\\crypto.h"
void PRNG_Open(CRYPTO_T *crypto, uint32_t u32KeySize, uint32_t u32SeedReload, uint32_t u32Seed);
void PRNG_Start(CRYPTO_T *crypto);
void PRNG_Read(CRYPTO_T *crypto, uint32_t u32RandKey[]);

void AES_Open(CRYPTO_T *crypto, uint32_t u32Channel, uint32_t u32EncDec, uint32_t u32OpMode, uint32_t u32KeySize, uint32_t u32SwapType);
void AES_Start(CRYPTO_T *crypto, uint32_t u32Channel, uint32_t u32DMAMode);
void AES_SetKey(CRYPTO_T *crypto, uint32_t u32Channel, uint32_t au32Keys[], uint32_t u32KeySize);
void AES_SetInitVect(CRYPTO_T *crypto, uint32_t u32Channel, uint32_t au32IV[]);
void AES_SetDMATransfer(CRYPTO_T *crypto, uint32_t u32Channel, uint32_t u32SrcAddr, uint32_t u32DstAddr, uint32_t u32TransCnt);
void AES_Start_KS(CRYPTO_T *crypto, uint32_t u32Channel, uint32_t u32DMAMode, int ksel, int knum);
void AES_SetKey_KS(CRYPTO_T *crypto, KS_MEM_Type mem, int32_t i32KeyIdx);

void CHA_SetKeyandNonce(CRYPTO_T *crypto, unsigned char *key, unsigned char *nonce, int counter);
void CHA_SetDMATransfer(CRYPTO_T *crypto, uint8_t *u8pInputData, uint8_t *u8pOutputData, int src_len);
void CHA_Start(CRYPTO_T *crypto, int is_encrypt);
void CHAPOLY_Complete(CRYPTO_T *crypto);
void CHAPOLY_Start(CRYPTO_T *crypto);
void POLY1305_SetKeyandClearNonce(CRYPTO_T *crypto, unsigned char *key);
void POLY1305_SetDMATransfer(CRYPTO_T *crypto, uint8_t *u8pInputData, uint8_t *u8pOutputData, int src_len);
void POLY1305_Start(CRYPTO_T *crypto);

void SHA_Open(CRYPTO_T *crypto, uint32_t u32OpMode, uint32_t u32SwapType, uint32_t hmac_key_len);
void SHA_Start(CRYPTO_T *crypto, uint32_t u32DMAMode);
void SHA_SetDMATransfer(CRYPTO_T *crypto, uint32_t u32SrcAddr, uint32_t u32TransCnt);
void SHA_Read(CRYPTO_T *crypto, uint32_t u32Digest[]);

void ECC_Enable_DFAP(void);
void ECC_Complete(CRYPTO_T *crypto);
int ECC_IsPrivateKeyValid(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char private_k[]);
int32_t ECC_GeneratePublicKey(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char *private_k, char public_k1[], char public_k2[]);
int32_t ECC_Mutiply(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char x1[], char y1[], char *k, char x2[], char y2[]);
int32_t ECC_GenerateSecretZ(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char *private_k, char public_k1[], char public_k2[], char secret_z[]);
int32_t ECC_GenerateSignature(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char *message, char *d, char *k, char *R, char *S);
int32_t ECC_VerifySignature(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char *message, char *public_k1, char *public_k2, char *R, char *S);
int32_t ECC_Write_N(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve);
int32_t ECC_GeneratePublicKey_KS(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, int k_ksnum, int is_ecdh, char public_k1[], char public_k2[]);
int32_t ECC_Mutiply_KS(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, int x1_ksnum, char x1[], int y1_ksnum, char y1[], int k_ksnum, char *k, char x2[], char y2[]);
int32_t ECC_GenerateSecretZ_KS(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, int k_ksnum, char *private_k, char public_x[], char public_y[], int z_to_ks, int z_owner, char secret_z[]);
int32_t ECC_GenerateSignature_KS(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char *message, int d_ksnum, int k_ksnum, char *R, char *S);
int32_t ECC_VerifySignature_KS(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char *message, int x_ksnum, int y_ksnum, char *R, char *S);
int32_t ECC_GenerateSignature_KS(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char *message, int d_ksnum, int k_ksnum, char *R, char *S);
int32_t ECC_GetCurve(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, ECC_CURVE *curve);
int32_t ECC_VerifySignature_KS(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char *message, int x_ksnum, int y_ksnum, char *R, char *S);
void Hex2Reg(char input[], uint32_t volatile reg[]);
void Reg2Hex(int32_t count, uint32_t volatile reg[], char output[]);
void Hex2RegEx(char input[], uint32_t volatile reg[], int shift);
int32_t ECC_GetCurve(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, ECC_CURVE *curve);
int ecc_strcmp(char *s1, char *s2);
int32_t ECC_VerifySignature_KS(CRYPTO_T *crypto, E_ECC_CURVE ecc_curve, char *message, int x_ksnum, int y_ksnum, char *R, char *S);
int32_t RSA_Open(CRYPTO_T *crypto, uint32_t u32OpMode, uint32_t u32KeySize, void *psRSA_Buf, uint32_t u32BufSize, uint32_t u32UseKS);
int32_t RSA_SetKey(CRYPTO_T *crypto, char *Key);
int32_t RSA_SetDMATransfer(CRYPTO_T *crypto, char *Src, char *n, char *P, char *Q);
void RSA_Start(CRYPTO_T *crypto);
int32_t RSA_Read(CRYPTO_T *crypto, char *Output);
int32_t RSA_SetKey_KS(CRYPTO_T *crypto, uint32_t u32KeyNum, uint32_t u32KSMemType, uint32_t u32BlindKeyNum);
int32_t RSA_SetDMATransfer_KS(CRYPTO_T *crypto, char *Src, char *n, uint32_t u32PNum,
                              uint32_t u32QNum, uint32_t u32CpNum, uint32_t u32CqNum, uint32_t u32DpNum,
                              uint32_t u32DqNum, uint32_t u32RpNum, uint32_t u32RqNum);
# 1711 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\kpi.h" 1
# 37 "../../../../Library/StdDriver/inc\\kpi.h"
typedef struct
{
    uint8_t x;
    uint8_t y;
    uint16_t st;
} KPI_KEY_T;
# 71 "../../../../Library/StdDriver/inc\\kpi.h"
int32_t KPI_Open(uint32_t u32Rows, uint32_t u32Columns, KPI_KEY_T *pkeyQueue, uint32_t u32MaxKeyCnt);
void KPI_Close(void);
void KPI_ConfigKeyScanTiming(uint32_t u32PreScale, uint32_t u32Debounce, uint32_t u32ScanDelay);
int32_t KPI_kbhit(void);
KPI_KEY_T KPI_GetKey(void);
# 1712 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\lpadc.h" 1
# 133 "../../../../Library/StdDriver/inc\\lpadc.h"
extern int32_t g_LPADC_i32ErrCode;
# 559 "../../../../Library/StdDriver/inc\\lpadc.h"
void LPADC_Open(LPADC_T *lpadc, uint32_t u32InputMode, uint32_t u32OpMode, uint32_t u32ChMask);
void LPADC_Calibration(LPADC_T *lpadc);
void LPADC_Close(LPADC_T *lpadc);
void LPADC_EnableHWTrigger(LPADC_T *lpadc, uint32_t u32Source, uint32_t u32Param);
void LPADC_DisableHWTrigger(LPADC_T *lpadc);
void LPADC_EnableInt(LPADC_T *lpadc, uint32_t u32Mask);
void LPADC_DisableInt(LPADC_T *lpadc, uint32_t u32Mask);
void LPADC_SetExtendSampleTime(LPADC_T *lpadc, uint32_t u32ModuleNum, uint32_t u32ExtendSampleTime);
void LPADC_SelectAutoOperationMode(LPADC_T *lpadc, uint32_t u32TrigSel);
# 1713 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\lpgpio.h" 1
# 67 "../../../../Library/StdDriver/inc\\lpgpio.h"
void LPGPIO_SetMode(LPGPIO_T *port, uint32_t u32PinMask, uint32_t u32Mode);
# 1714 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\lpi2c.h" 1
# 80 "../../../../Library/StdDriver/inc\\lpi2c.h"
extern int32_t g_LPI2C_i32ErrCode;
# 262 "../../../../Library/StdDriver/inc\\lpi2c.h"
static __inline void LPI2C_STOP(LPI2C_T *i2c);
# 271 "../../../../Library/StdDriver/inc\\lpi2c.h"
static __inline void LPI2C_STOP(LPI2C_T *i2c)
{

    (i2c)->CTL0 |= ((0x1ul << (3)) | (0x1ul << (4)));

    while (i2c->CTL0 & (0x1ul << (4)))
    {
    }
}

void LPI2C_ClearTimeoutFlag(LPI2C_T *i2c);
void LPI2C_Close(LPI2C_T *i2c);
void LPI2C_Trigger(LPI2C_T *i2c, uint8_t u8Start, uint8_t u8Stop, uint8_t u8Si, uint8_t u8Ack);
void LPI2C_DisableInt(LPI2C_T *i2c);
void LPI2C_EnableInt(LPI2C_T *i2c);
uint32_t LPI2C_GetBusClockFreq(LPI2C_T *i2c);
uint32_t LPI2C_GetIntFlag(LPI2C_T *i2c);
uint32_t LPI2C_GetStatus(LPI2C_T *i2c);
uint32_t LPI2C_Open(LPI2C_T *i2c, uint32_t u32BusClock);
uint8_t LPI2C_GetData(LPI2C_T *i2c);
void LPI2C_SetSlaveAddr(LPI2C_T *i2c, uint8_t u8SlaveNo, uint8_t u8SlaveAddr, uint8_t u8GCMode);
void LPI2C_SetSlaveAddrMask(LPI2C_T *i2c, uint8_t u8SlaveNo, uint8_t u8SlaveAddrMask);
uint32_t LPI2C_SetBusClockFreq(LPI2C_T *i2c, uint32_t u32BusClock);
void LPI2C_EnableTimeout(LPI2C_T *i2c, uint8_t u8LongTimeout);
void LPI2C_DisableTimeout(LPI2C_T *i2c);
void LPI2C_EnableWakeup(LPI2C_T *i2c);
void LPI2C_DisableWakeup(LPI2C_T *i2c);
void LPI2C_SetData(LPI2C_T *i2c, uint8_t u8Data);
uint8_t LPI2C_WriteByte(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint8_t data);
uint32_t LPI2C_WriteMultiBytes(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint8_t data[], uint32_t u32wLen);
uint8_t LPI2C_WriteByteOneReg(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr, uint8_t data);
uint32_t LPI2C_WriteMultiBytesOneReg(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr, uint8_t data[], uint32_t u32wLen);
uint8_t LPI2C_WriteByteTwoRegs(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr, uint8_t data);
uint32_t LPI2C_WriteMultiBytesTwoRegs(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr, uint8_t data[], uint32_t u32wLen);
uint8_t LPI2C_ReadByte(LPI2C_T *i2c, uint8_t u8SlaveAddr);
uint32_t LPI2C_ReadMultiBytes(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint8_t rdata[], uint32_t u32rLen);
uint8_t LPI2C_ReadByteOneReg(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr);
uint32_t LPI2C_ReadMultiBytesOneReg(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr, uint8_t rdata[], uint32_t u32rLen);
uint8_t LPI2C_ReadByteTwoRegs(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr);
uint32_t LPI2C_ReadMultiBytesTwoRegs(LPI2C_T *i2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr, uint8_t rdata[], uint32_t u32rLen);
# 1715 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\lppdma.h" 1
# 267 "../../../../Library/StdDriver/inc\\lppdma.h"
void LPPDMA_Open(LPPDMA_T *lppdma, uint32_t u32Mask);
void LPPDMA_Close(LPPDMA_T *lppdma);
void LPPDMA_SetTransferCnt(LPPDMA_T *lppdma, uint32_t u32Ch, uint32_t u32Width, uint32_t u32TransCount);
void LPPDMA_SetTransferAddr(LPPDMA_T *lppdma, uint32_t u32Ch, uint32_t u32SrcAddr, uint32_t u32SrcCtrl, uint32_t u32DstAddr, uint32_t u32DstCtrl);
void LPPDMA_SetTransferMode(LPPDMA_T *lppdma, uint32_t u32Ch, uint32_t u32Peripheral, uint32_t u32ScatterEn, uint32_t u32DescAddr);
void LPPDMA_SetBurstType(LPPDMA_T *lppdma, uint32_t u32Ch, uint32_t u32BurstType, uint32_t u32BurstSize);
void LPPDMA_Trigger(LPPDMA_T *lppdma, uint32_t u32Ch);
void LPPDMA_EnableInt(LPPDMA_T *lppdma, uint32_t u32Ch, uint32_t u32Mask);
void LPPDMA_DisableInt(LPPDMA_T *lppdma, uint32_t u32Ch, uint32_t u32Mask);
# 1716 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\lpspi.h" 1
# 579 "../../../../Library/StdDriver/inc\\lpspi.h"
uint32_t LPSPI_Open(LPSPI_T *lpspi, uint32_t u32MasterSlave, uint32_t u32SPIMode, uint32_t u32DataWidth, uint32_t u32BusClock);
void LPSPI_Close(LPSPI_T *lpspi);
void LPSPI_ClearRxFIFO(LPSPI_T *lpspi);
void LPSPI_ClearTxFIFO(LPSPI_T *lpspi);
void LPSPI_DisableAutoSS(LPSPI_T *lpspi);
void LPSPI_EnableAutoSS(LPSPI_T *lpspi, uint32_t u32SSPinMask, uint32_t u32ActiveLevel);
uint32_t LPSPI_SetBusClock(LPSPI_T *lpspi, uint32_t u32BusClock);
void LPSPI_SetFIFO(LPSPI_T *lpspi, uint32_t u32TxThreshold, uint32_t u32RxThreshold);
uint32_t LPSPI_GetBusClock(LPSPI_T *lpspi);
void LPSPI_EnableInt(LPSPI_T *lpspi, uint32_t u32Mask);
void LPSPI_DisableInt(LPSPI_T *lpspi, uint32_t u32Mask);
uint32_t LPSPI_GetIntFlag(LPSPI_T *lpspi, uint32_t u32Mask);
void LPSPI_ClearIntFlag(LPSPI_T *lpspi, uint32_t u32Mask);
uint32_t LPSPI_GetStatus(LPSPI_T *lpspi, uint32_t u32Mask);
# 1717 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\lptmr.h" 1
# 181 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_Start(LPTMR_T *lptmr);
static __inline void LPTMR_Stop(LPTMR_T *lptmr);
static __inline void LPTMR_EnableWakeup(LPTMR_T *lptmr);
static __inline void LPTMR_DisableWakeup(LPTMR_T *lptmr);
static __inline void LPTMR_StartCapture(LPTMR_T *lptmr);
static __inline void LPTMR_StopCapture(LPTMR_T *lptmr);
static __inline void LPTMR_EnableCaptureDebounce(LPTMR_T *lptmr);
static __inline void LPTMR_DisableCaptureDebounce(LPTMR_T *lptmr);
static __inline void LPTMR_EnableEventCounterDebounce(LPTMR_T *lptmr);
static __inline void LPTMR_DisableEventCounterDebounce(LPTMR_T *lptmr);
static __inline void LPTMR_EnableInt(LPTMR_T *lptmr);
static __inline void LPTMR_DisableInt(LPTMR_T *lptmr);
static __inline void LPTMR_EnableCaptureInt(LPTMR_T *lptmr);
static __inline void LPTMR_DisableCaptureInt(LPTMR_T *lptmr);
static __inline uint32_t LPTMR_GetIntFlag(LPTMR_T *lptmr);
static __inline void LPTMR_ClearIntFlag(LPTMR_T *lptmr);
static __inline uint32_t LPTMR_GetCaptureIntFlag(LPTMR_T *lptmr);
static __inline uint32_t LPTMR_GetCaptureIntFlagOV(LPTMR_T *lptmr);
static __inline void LPTMR_ClearCaptureIntFlag(LPTMR_T *lptmr);
static __inline uint32_t LPTMR_GetWakeupFlag(LPTMR_T *lptmr);
static __inline void LPTMR_ClearWakeupFlag(LPTMR_T *lptmr);
static __inline uint32_t LPTMR_GetCaptureData(LPTMR_T *lptmr);
static __inline uint32_t LPTMR_GetCounter(LPTMR_T *lptmr);
static __inline void LPTMR_EnablePDCLK(LPTMR_T *lptmr);
static __inline void LPTMR_DisablePDCLK(LPTMR_T *lptmr);
static __inline void LPTMR_EventCounterSelect(LPTMR_T *lptmr, uint32_t u32Src);
# 217 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_Start(LPTMR_T *lptmr)
{
    lptmr->CTL |= (0x1UL << (30));
}
# 231 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_Stop(LPTMR_T *lptmr)
{
    lptmr->CTL &= ~(0x1UL << (30));
}
# 247 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_EnableWakeup(LPTMR_T *lptmr)
{
    lptmr->CTL |= ((0x1UL << (23)) | (0x1UL << (16)));
}
# 261 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_DisableWakeup(LPTMR_T *lptmr)
{
    lptmr->CTL &= ~(0x1UL << (23));
}
# 275 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_StartCapture(LPTMR_T *lptmr)
{
    lptmr->EXTCTL |= (0x1UL << (3));
}
# 289 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_StopCapture(LPTMR_T *lptmr)
{
    lptmr->EXTCTL &= ~(0x1UL << (3));
}
# 303 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_EnableCaptureDebounce(LPTMR_T *lptmr)
{
    lptmr->EXTCTL |= (0x1UL << (6));
}
# 317 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_DisableCaptureDebounce(LPTMR_T *lptmr)
{
    lptmr->EXTCTL &= ~(0x1UL << (6));
}
# 331 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_EnableEventCounterDebounce(LPTMR_T *lptmr)
{
    lptmr->EXTCTL |= (0x1UL << (7));
}
# 345 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_DisableEventCounterDebounce(LPTMR_T *lptmr)
{
    lptmr->EXTCTL &= ~(0x1UL << (7));
}
# 359 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_EnableInt(LPTMR_T *lptmr)
{
    lptmr->CTL |= (0x1UL << (29));
}
# 373 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_DisableInt(LPTMR_T *lptmr)
{
    lptmr->CTL &= ~(0x1UL << (29));
}
# 387 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_EnableCaptureInt(LPTMR_T *lptmr)
{
    lptmr->EXTCTL |= (0x1UL << (5));
}
# 401 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_DisableCaptureInt(LPTMR_T *lptmr)
{
    lptmr->EXTCTL &= ~(0x1UL << (5));
}
# 416 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline uint32_t LPTMR_GetIntFlag(LPTMR_T *lptmr)
{
    return ((lptmr->INTSTS & (0x1UL << (0))) ? 1UL : 0UL);
}
# 430 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_ClearIntFlag(LPTMR_T *lptmr)
{
    lptmr->INTSTS = (0x1UL << (0));
}
# 445 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline uint32_t LPTMR_GetCaptureIntFlag(LPTMR_T *lptmr)
{
    return ((lptmr->EINTSTS & (0x1UL << (0))) ? 1UL : 0UL);
}
# 460 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline uint32_t LPTMR_GetCaptureIntFlagOV(LPTMR_T *lptmr)
{
    return ((lptmr->EINTSTS & (0x1UL << (1))) ? 1UL : 0UL);
}
# 474 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_ClearCaptureIntFlag(LPTMR_T *lptmr)
{
    lptmr->EINTSTS = (0x1UL << (0));
}
# 489 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline uint32_t LPTMR_GetWakeupFlag(LPTMR_T *lptmr)
{
    return (lptmr->INTSTS & (0x1UL << (1)) ? 1UL : 0UL);
}
# 503 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_ClearWakeupFlag(LPTMR_T *lptmr)
{
    lptmr->INTSTS = (0x1UL << (1));
}
# 517 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline uint32_t LPTMR_GetCaptureData(LPTMR_T *lptmr)
{
    return lptmr->CAP;
}
# 531 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline uint32_t LPTMR_GetCounter(LPTMR_T *lptmr)
{
    return lptmr->CNT;
}
# 545 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_EnablePDCLK(LPTMR_T *lptmr)
{
    lptmr->CTL |= (0x1UL << (16));
}
# 559 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_DisablePDCLK(LPTMR_T *lptmr)
{
    lptmr->CTL &= ~(0x1UL << (16));
}
# 578 "../../../../Library/StdDriver/inc\\lptmr.h"
static __inline void LPTMR_EventCounterSelect(LPTMR_T *lptmr, uint32_t u32Src)
{
    lptmr->EXTCTL = (lptmr->EXTCTL & ~(0x7UL << (16))) | u32Src;
}

uint32_t LPTMR_Open(LPTMR_T *lptmr, uint32_t u32Mode, uint32_t u32Freq);
void LPTMR_Close(LPTMR_T *lptmr);
int32_t LPTMR_Delay(LPTMR_T *lptmr, uint32_t u32Usec);
void LPTMR_EnableCapture(LPTMR_T *lptmr, uint32_t u32CapMode, uint32_t u32Edge);
void LPTMR_CaptureSelect(LPTMR_T *lptmr, uint32_t u32Src);
void LPTMR_DisableCapture(LPTMR_T *lptmr);
void LPTMR_EnableEventCounter(LPTMR_T *lptmr, uint32_t u32Edge);
void LPTMR_DisableEventCounter(LPTMR_T *lptmr);
uint32_t LPTMR_GetModuleClock(LPTMR_T *lptmr);
void LPTMR_SetTriggerSource(LPTMR_T *lptmr, uint32_t u32Src);
void LPTMR_SetTriggerTarget(LPTMR_T *lptmr, uint32_t u32Mask);
int32_t LPTMR_ResetCounter(LPTMR_T *lptmr);
void LPTMR_EnableCaptureInputNoiseFilter(LPTMR_T *lptmr, uint32_t u32FilterCount, uint32_t u32ClkSrcSel);
void LPTMR_DisableCaptureInputNoiseFilter(LPTMR_T *lptmr);
# 1718 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\lptmr_pwm.h" 1
# 438 "../../../../Library/StdDriver/inc\\lptmr_pwm.h"
uint32_t LPTPWM_ConfigOutputFreqAndDuty(LPTMR_T *lptmr, uint32_t u32Frequency, uint32_t u32DutyCycle);
void LPTPWM_EnableCounter(LPTMR_T *lptmr);
void LPTPWM_DisableCounter(LPTMR_T *lptmr);
void LPTPWM_EnableTrigger(LPTMR_T *lptmr, uint32_t u32TargetMask, uint32_t u32Condition);
void LPTPWM_DisableTrigger(LPTMR_T *lptmr, uint32_t u32TargetMask);


static __inline void LPTPWM_EnableWakeup(LPTMR_T *lptmr);
static __inline void LPTPWM_DisableWakeup(LPTMR_T *lptmr);
static __inline uint32_t LPTPWM_GetWakeupFlag(LPTMR_T *lptmr);
static __inline void LPTPWM_ClearWakeupFlag(LPTMR_T *lptmr);
static __inline void LPTPWM_EnablePDCLK(LPTMR_T *lptmr);
static __inline void LPTPWM_DisablePDCLK(LPTMR_T *lptmr);
static __inline void LPTPWM_EnableAcc(LPTMR_T *lptmr, uint32_t u32IntFlagCnt, uint32_t u32IntAccSrc);
static __inline void LPTPWM_DisableAcc(LPTMR_T *lptmr);
static __inline void LPTPWM_EnableAccInt(LPTMR_T *lptmr);
static __inline void LPTPWM_DisableAccInt(LPTMR_T *lptmr);
static __inline void LPTPWM_ClearAccInt(LPTMR_T *lptmr);
static __inline uint32_t LPTPWM_GetAccInt(LPTMR_T *lptmr);
static __inline void LPTPWM_EnableAccLPPDMA(LPTMR_T *lptmr);
static __inline void LPTPWM_DisableAccLPPDMA(LPTMR_T *lptmr);
static __inline void LPTPWM_EnableAccStopMode(LPTMR_T *lptmr);
static __inline void LPTPWM_DisableAccStopMode(LPTMR_T *lptmr);
# 474 "../../../../Library/StdDriver/inc\\lptmr_pwm.h"
static __inline void LPTPWM_EnableWakeup(LPTMR_T *lptmr)
{
    lptmr->CTL |= (0x1UL << (16));
    lptmr->PWMCTL |= (0x1UL << (12));
}
# 490 "../../../../Library/StdDriver/inc\\lptmr_pwm.h"
static __inline void LPTPWM_DisableWakeup(LPTMR_T *lptmr)
{
    lptmr->PWMCTL &= ~(0x1UL << (12));
}
# 506 "../../../../Library/StdDriver/inc\\lptmr_pwm.h"
static __inline uint32_t LPTPWM_GetWakeupFlag(LPTMR_T *lptmr)
{
    return ((lptmr->PWMSTATUS & (0x1UL << (8))) ? 1 : 0);
}
# 521 "../../../../Library/StdDriver/inc\\lptmr_pwm.h"
static __inline void LPTPWM_ClearWakeupFlag(LPTMR_T *lptmr)
{
    lptmr->PWMSTATUS = (0x1UL << (8));
}
# 535 "../../../../Library/StdDriver/inc\\lptmr_pwm.h"
static __inline void LPTPWM_EnablePDCLK(LPTMR_T *lptmr)
{
    lptmr->CTL |= (0x1UL << (16));
}
# 549 "../../../../Library/StdDriver/inc\\lptmr_pwm.h"
static __inline void LPTPWM_DisablePDCLK(LPTMR_T *lptmr)
{
    lptmr->CTL &= ~(0x1UL << (16));
}
# 564 "../../../../Library/StdDriver/inc\\lptmr_pwm.h"
static __inline void LPTPWM_EnableAcc(LPTMR_T *lptmr, uint32_t u32IntFlagCnt, uint32_t u32IntAccSrc)
{
    lptmr->PWMIFA = (((lptmr)->PWMIFA & ~((0xffffUL << (0)) | (0x3UL << (28)) | (0x1UL << (24))))
                     | ((0x1UL << (31)) | (u32IntFlagCnt << (0)) | (u32IntAccSrc << (28))));
}







static __inline void LPTPWM_DisableAcc(LPTMR_T *lptmr)
{
    lptmr->PWMIFA &= ~(0x1UL << (31));
}







static __inline void LPTPWM_EnableAccInt(LPTMR_T *lptmr)
{
    lptmr->PWMAINTEN |= (0x1UL << (0));
}







static __inline void LPTPWM_DisableAccInt(LPTMR_T *lptmr)
{
    lptmr->PWMAINTEN &= ~(0x1UL << (0));
}







static __inline void LPTPWM_ClearAccInt(LPTMR_T *lptmr)
{
    lptmr->PWMAINTSTS = (0x1UL << (0));
}
# 621 "../../../../Library/StdDriver/inc\\lptmr_pwm.h"
static __inline uint32_t LPTPWM_GetAccInt(LPTMR_T *lptmr)
{
    return (((lptmr)->PWMAINTSTS & (0x1UL << (0))) ? 1UL : 0UL);
}







static __inline void LPTPWM_EnableAccLPPDMA(LPTMR_T *lptmr)
{
    lptmr->PWMAPDMACTL |= (0x1UL << (0));
}







static __inline void LPTPWM_DisableAccLPPDMA(LPTMR_T *lptmr)
{
    lptmr->PWMAPDMACTL &= ~(0x1UL << (0));
}







static __inline void LPTPWM_EnableAccStopMode(LPTMR_T *lptmr)
{
    lptmr->PWMIFA |= (0x1UL << (24));
}







static __inline void LPTPWM_DisableAccStopMode(LPTMR_T *lptmr)
{
    lptmr->PWMIFA &= ~(0x1UL << (24));
}

uint32_t LPTPWM_ConfigOutputFreqAndDuty(LPTMR_T *lptmr, uint32_t u32Frequency, uint32_t u32DutyCycle);
void LPTPWM_EnableCounter(LPTMR_T *lptmr);
void LPTPWM_DisableCounter(LPTMR_T *lptmr);
void LPTPWM_EnableTrigger(LPTMR_T *lptmr, uint32_t u32TargetMask, uint32_t u32Condition);
void LPTPWM_DisableTrigger(LPTMR_T *lptmr, uint32_t u32TargetMask);
# 1719 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\lpuart.h" 1
# 418 "../../../../Library/StdDriver/inc\\lpuart.h"
static __inline void LPUART_CLEAR_RTS(LPUART_T *lpuart);
static __inline void LPUART_SET_RTS(LPUART_T *lpuart);
# 431 "../../../../Library/StdDriver/inc\\lpuart.h"
static __inline void LPUART_CLEAR_RTS(LPUART_T *lpuart)
{
    lpuart->MODEM |= (0x1ul << (9));
    lpuart->MODEM &= ~(0x1ul << (1));
}
# 447 "../../../../Library/StdDriver/inc\\lpuart.h"
static __inline void LPUART_SET_RTS(LPUART_T *lpuart)
{
    lpuart->MODEM |= (0x1ul << (9)) | (0x1ul << (1));
}
# 639 "../../../../Library/StdDriver/inc\\lpuart.h"
void LPUART_ClearIntFlag(LPUART_T *lpuart, uint32_t u32InterruptFlag);
void LPUART_Close(LPUART_T *lpuart);
void LPUART_DisableFlowCtrl(LPUART_T *lpuart);
void LPUART_DisableInt(LPUART_T *lpuart, uint32_t u32InterruptFlag);
void LPUART_EnableFlowCtrl(LPUART_T *lpuart);
void LPUART_EnableInt(LPUART_T *lpuart, uint32_t u32InterruptFlag);
void LPUART_Open(LPUART_T *lpuart, uint32_t u32baudrate);
uint32_t LPUART_Read(LPUART_T *lpuart, uint8_t pu8RxBuf[], uint32_t u32ReadBytes);
void LPUART_SetLineConfig(LPUART_T *lpuart, uint32_t u32baudrate, uint32_t u32data_width, uint32_t u32parity, uint32_t u32stop_bits);
void LPUART_SetTimeoutCnt(LPUART_T *lpuart, uint32_t u32TOC);
void LPUART_SelectRS485Mode(LPUART_T *lpuart, uint32_t u32Mode, uint32_t u32Addr);
uint32_t LPUART_Write(LPUART_T *lpuart, uint8_t pu8TxBuf[], uint32_t u32WriteBytes);
void LPUART_SelectAutoOperationMode(LPUART_T *lpuart, uint32_t u32TrigSel);
# 1720 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\otfc.h" 1
# 411 "../../../../Library/StdDriver/inc\\otfc.h"
int32_t OTFC_SetKeyFromKeyReg(OTFC_T *otfc,
                              uint32_t *pau32KeyTable,
                              uint32_t u32PR,
                              uint32_t u32SAddr,
                              uint32_t u32PRSize);

int32_t OTFC_SetKeyFromKeyStore(OTFC_T *otfc, uint32_t u32PR,
                                uint32_t u32SAddr, uint32_t u32PRSize,
                                uint32_t u32KeyNum, uint32_t u32KeySrc);

int32_t OTFC_SetScrambleNum(OTFC_T *otfc, uint32_t u32PR, uint32_t u32Scramble);
int32_t OTFC_SetNonceNum(OTFC_T *otfc, uint32_t u32PR, uint32_t u32Nonce0, uint32_t u32Nonce1, uint32_t u32Nonce2);
# 1721 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\otg.h" 1
# 1722 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\pdma.h" 1
# 365 "../../../../Library/StdDriver/inc\\pdma.h"
void PDMA_Open(PDMA_T *pdma, uint32_t u32Mask);
void PDMA_Close(PDMA_T *pdma);
void PDMA_SetTransferCnt(PDMA_T *pdma, uint32_t u32Ch, uint32_t u32Width, uint32_t u32TransCount);
void PDMA_SetTransferAddr(PDMA_T *pdma, uint32_t u32Ch, uint32_t u32SrcAddr, uint32_t u32SrcCtrl, uint32_t u32DstAddr, uint32_t u32DstCtrl);
void PDMA_SetTransferMode(PDMA_T *pdma, uint32_t u32Ch, uint32_t u32Peripheral, uint32_t u32ScatterEn, uint32_t u32DescAddr);
void PDMA_SetBurstType(PDMA_T *pdma, uint32_t u32Ch, uint32_t u32BurstType, uint32_t u32BurstSize);
void PDMA_EnableTimeout(PDMA_T *pdma, uint32_t u32Mask);
void PDMA_DisableTimeout(PDMA_T *pdma, uint32_t u32Mask);
void PDMA_SetTimeOut(PDMA_T *pdma, uint32_t u32Ch, uint32_t u32OnOff, uint32_t u32TimeOutCnt);
void PDMA_Trigger(PDMA_T *pdma, uint32_t u32Ch);
void PDMA_EnableInt(PDMA_T *pdma, uint32_t u32Ch, uint32_t u32Mask);
void PDMA_DisableInt(PDMA_T *pdma, uint32_t u32Ch, uint32_t u32Mask);
# 1723 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\pmc.h" 1
# 30 "../../../../Library/StdDriver/inc\\pmc.h"
enum
{
    SRAMNum0 = 0,
    SRAMNum1,
    SRAMNum2,
    SRAMNum3,
    SRAMNum4,
};
# 266 "../../../../Library/StdDriver/inc\\pmc.h"
int32_t PMC_SetPowerLevel(uint32_t u32PowerLevel);
int32_t PMC_SetPowerRegulator(uint32_t u32PowerRegulator);
int32_t PMC_SetSRAMPowerMode(uint32_t u32SRAMSel, uint32_t u32PowerMode);
void PMC_PowerDown(void);
void PMC_Idle(void);
int32_t PMC_SetPowerDownMode(uint32_t u32PDMode, uint32_t u32PowerLevel);
void PMC_EnableWKPIN(uint32_t u32TriggerType);
uint32_t PMC_GetPMCWKSrc(void);
void PMC_EnableTGPin(uint32_t u32Port, uint32_t u32Pin, uint32_t u32TriggerType, uint32_t u32DebounceEn, uint32_t u32WakeupEn);
int32_t PMC_EnableSTMR(uint32_t u32Interval);
int32_t PMC_DisableSTMR(void);
int32_t PMC_Wait_BusyFlag(uint32_t PMCBusyFlagAddr);
# 1724 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\psio.h" 1
# 118 "../../../../Library/StdDriver/inc\\psio.h"
typedef struct
{
unsigned CKPT0SLT:
    4;
    unsigned CKPT1SLT: 4;
    unsigned CKPT2SLT: 4;
    unsigned CKPT3SLT: 4;
    unsigned CKPT4SLT: 4;
    unsigned CKPT5SLT: 4;
    unsigned CKPT6SLT: 4;
    unsigned CKPT7SLT: 4;
unsigned CKPT0ACT:
    4;
    unsigned CKPT1ACT: 4;
    unsigned CKPT2ACT: 4;
    unsigned CKPT3ACT: 4;
    unsigned CKPT4ACT: 4;
    unsigned CKPT5ACT: 4;
    unsigned CKPT6ACT: 4;
    unsigned CKPT7ACT: 4;
} S_PSIO_CP_CONFIG;
# 961 "../../../../Library/StdDriver/inc\\psio.h"
static __inline void PSIO_SET_INTCTL(PSIO_T *psio, uint32_t u32SC, uint32_t u32Int, uint32_t u32Slot)
{
    if (u32Int == 0x00000000UL)
    {
        (psio)->INTCTL = (((psio)->INTCTL & ~(0xful << (0)) & ~(0x3ul << (8)))
                          | ((u32SC) << (8))
                          | ((u32Slot) << (0)));
    }
    else if (u32Int == 0x00000001UL)
    {
        (psio)->INTCTL = (((psio)->INTCTL & ~(0xful << (4)) & ~(0x3ul << (12)))
                          | ((u32SC) << (12))
                          | ((u32Slot) << (4)));
    }
}
# 990 "../../../../Library/StdDriver/inc\\psio.h"
static __inline void PSIO_CLEAR_INTCTL(PSIO_T *psio, uint32_t u32Int)
{
    if (u32Int == 0x00000000UL)
    {
        (psio)->INTCTL = ((psio)->INTCTL & ~(0xful << (0)) & ~(0x3ul << (8)));
    }
    else if (u32Int == 0x00000001UL)
    {
        (psio)->INTCTL = ((psio)->INTCTL & ~(0xful << (4)) & ~(0x3ul << (12)));
    }
}
# 1038 "../../../../Library/StdDriver/inc\\psio.h"
static __inline void PSIO_SET_SCCTL(PSIO_T *psio, uint32_t u32SC, uint32_t u32InitSlot, uint32_t u32EndSlot, uint32_t u32LoopCnt, uint32_t u32Repeat)
{
    (psio)->SCCT[u32SC].SCCTL = ((psio)->SCCT[u32SC].SCCTL & ~(0xful << (0)) & ~(0xful << (4)) & ~(0x3ful << (8)))
                                | ((u32InitSlot) << (0))
                                | ((u32EndSlot) << (4))
                                | ((u32LoopCnt & 0x3F) << (8));

    if (u32Repeat == 0x00000001UL)
        (psio)->SCCT[u32SC].SCCTL |= (0x1ul << (17));
    else if (u32Repeat == 0x00000000UL)
        (psio)->SCCT[u32SC].SCCTL &= ~(0x1ul << (17));
}
# 1093 "../../../../Library/StdDriver/inc\\psio.h"
static __inline void PSIO_SET_GENCTL(PSIO_T *psio, uint32_t u32Pin, uint32_t u32PinEn, uint32_t u32SC, uint32_t u32IOMode, uint32_t u32PinInit, uint32_t u32PinInterval)
{
    (psio)->GNCT[u32Pin].GENCTL = ((psio)->GNCT[u32Pin].GENCTL & ~(0x3ul << (24)) & ~(0x3ul << (0))
                                   & ~(0x3ul << (2)) & ~(0x3ul << (4)))
                                  | ((u32SC) << (24)) | ((u32IOMode) << (0))
                                  | ((u32PinInit) << (2)) | ((u32PinInterval) << (4));

    if (u32PinEn == 0x00000001UL)
        (psio)->GNCT[u32Pin].GENCTL |= (0x1ul << (26));
    else if (u32PinEn == 0x00000000UL)
        (psio)->GNCT[u32Pin].GENCTL &= ~(0x1ul << (26));
}
# 1141 "../../../../Library/StdDriver/inc\\psio.h"
static __inline void PSIO_SWITCH_MODE(PSIO_T *psio, uint32_t u32Pin, uint32_t u32SwPoint, uint32_t u32SwMode, uint32_t u32SwCP)
{
    if (u32SwPoint == 0x00000000UL)
    {
        (psio)->GNCT[u32Pin].GENCTL = ((psio)->GNCT[u32Pin].GENCTL & ~(0x3ul << (16)) & ~(0xful << (8)))
                                      | ((u32SwMode) << (16)) | ((u32SwCP + 1) << (8));
    }
    else if (u32SwPoint == 0x00000001UL)
    {
        (psio)->GNCT[u32Pin].GENCTL = ((psio)->GNCT[u32Pin].GENCTL & ~(0x3ul << (18)) & ~(0xful << (12)))
                                      | ((u32SwMode) << (18)) | ((u32SwCP + 1) << (12));
    }
}
# 1175 "../../../../Library/StdDriver/inc\\psio.h"
static __inline void PSIO_SET_CP_CONFIG(PSIO_T *psio, uint32_t u32Pin, const S_PSIO_CP_CONFIG *sConfig)
{
    psio->GNCT[u32Pin].CPCTL0 = *(uint32_t *)sConfig;
    psio->GNCT[u32Pin].CPCTL1 = *((uint32_t *)sConfig + 1);
}
# 1725 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\qspi.h" 1
# 399 "../../../../Library/StdDriver/inc\\qspi.h"
uint32_t QSPI_Open(QSPI_T *qspi, uint32_t u32MasterSlave, uint32_t u32QSPIMode, uint32_t u32DataWidth, uint32_t u32BusClock);
void QSPI_Close(QSPI_T *qspi);
void QSPI_ClearRxFIFO(QSPI_T *qspi);
void QSPI_ClearTxFIFO(QSPI_T *qspi);
void QSPI_DisableAutoSS(QSPI_T *qspi);
void QSPI_EnableAutoSS(QSPI_T *qspi, uint32_t u32SSPinMask, uint32_t u32ActiveLevel);
uint32_t QSPI_SetBusClock(QSPI_T *qspi, uint32_t u32BusClock);
void QSPI_SetFIFO(QSPI_T *qspi, uint32_t u32TxThreshold, uint32_t u32RxThreshold);
uint32_t QSPI_GetBusClock(QSPI_T *qspi);
void QSPI_EnableInt(QSPI_T *qspi, uint32_t u32Mask);
void QSPI_DisableInt(QSPI_T *qspi, uint32_t u32Mask);
uint32_t QSPI_GetIntFlag(QSPI_T *qspi, uint32_t u32Mask);
void QSPI_ClearIntFlag(QSPI_T *qspi, uint32_t u32Mask);
uint32_t QSPI_GetStatus(QSPI_T *qspi, uint32_t u32Mask);
# 1726 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\rng.h" 1
# 37 "../../../../Library/StdDriver/inc\\rng.h"
int32_t RNG_Open(void);
int32_t RNG_Random(uint32_t *pu32Buf, int32_t nWords);
int32_t RNG_ECDSA_Init(uint32_t u32KeySize, uint32_t au32ECC_N[18]);
int32_t RNG_ECDSA(uint32_t u32KeySize);
int32_t RNG_ECDH_Init(uint32_t u32KeySize, uint32_t au32ECC_N[18]);
int32_t RNG_ECDH(uint32_t u32KeySize);
int32_t RNG_EntropyPoll(uint32_t *pu32Out, int32_t i32Len);
# 1727 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\rtc.h" 1
# 146 "../../../../Library/StdDriver/inc\\rtc.h"
typedef struct
{
    uint32_t u32Year;
    uint32_t u32Month;
    uint32_t u32Day;
    uint32_t u32DayOfWeek;
    uint32_t u32Hour;
    uint32_t u32Minute;
    uint32_t u32Second;
    uint32_t u32TimeScale;
    uint32_t u32AmPm;
} S_RTC_TIME_DATA_T;
# 431 "../../../../Library/StdDriver/inc\\rtc.h"
int32_t RTC_Open(S_RTC_TIME_DATA_T *sPt);
void RTC_Close(void);
int32_t RTC_32KCalibration(int32_t i32FrequencyX10000);
void RTC_GetDateAndTime(S_RTC_TIME_DATA_T *sPt);
void RTC_GetAlarmDateAndTime(S_RTC_TIME_DATA_T *sPt);
void RTC_SetDateAndTime(S_RTC_TIME_DATA_T *sPt);
void RTC_SetAlarmDateAndTime(S_RTC_TIME_DATA_T *sPt);
void RTC_SetDate(uint32_t u32Year, uint32_t u32Month, uint32_t u32Day, uint32_t u32DayOfWeek);
void RTC_SetTime(uint32_t u32Hour, uint32_t u32Minute, uint32_t u32Second, uint32_t u32TimeMode, uint32_t u32AmPm);
void RTC_SetAlarmDate(uint32_t u32Year, uint32_t u32Month, uint32_t u32Day);
void RTC_SetAlarmTime(uint32_t u32Hour, uint32_t u32Minute, uint32_t u32Second, uint32_t u32TimeMode, uint32_t u32AmPm);
void RTC_SetAlarmDateMask(uint8_t u8IsTenYMsk, uint8_t u8IsYMsk, uint8_t u8IsTenMMsk, uint8_t u8IsMMsk, uint8_t u8IsTenDMsk, uint8_t u8IsDMsk);
void RTC_SetAlarmTimeMask(uint8_t u8IsTenHMsk, uint8_t u8IsHMsk, uint8_t u8IsTenMMsk, uint8_t u8IsMMsk, uint8_t u8IsTenSMsk, uint8_t u8IsSMsk);
uint32_t RTC_GetDayOfWeek(void);
void RTC_SetTickPeriod(uint32_t u32TickSelection);
void RTC_EnableInt(uint32_t u32IntFlagMask);
void RTC_DisableInt(uint32_t u32IntFlagMask);
void RTC_EnableSpareAccess(void);
void RTC_DisableSpareRegister(void);
void RTC_StaticTamperEnable(uint32_t u32TamperSelect, uint32_t u32DetecLevel, uint32_t u32DebounceEn);
void RTC_StaticTamperDisable(uint32_t u32TamperSelect);
void RTC_DynamicTamperEnable(uint32_t u32PairSel, uint32_t u32DebounceEn, uint32_t u32Pair1Source, uint32_t u32Pair2Source);
void RTC_DynamicTamperDisable(uint32_t u32PairSel);
void RTC_DynamicTamperConfig(uint32_t u32ChangeRate, uint32_t u32SeedReload, uint32_t u32RefPattern, uint32_t u32Seed);
uint32_t RTC_SetClockSource(uint32_t u32ClkSrc);
void RTC_SetGPIOMode(uint32_t u32PFPin, uint32_t u32Mode, uint32_t u32DigitalCtl, uint32_t u32PullCtl, uint32_t u32OutputLevel);
void RTC_SetGPIOLevel(uint32_t u32PFPin, uint32_t u32OutputLevel);
void RTC_EnableClockFrequencyDetector(uint32_t u32FailBoundary, uint32_t u32StopBoundary);
void RTC_DisableClockFrequencyDetector(void);
# 1728 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\sc.h" 1
# 210 "../../../../Library/StdDriver/inc\\sc.h"
static __inline void SC_SetTxRetry(SC_T *sc, uint32_t u32Count);
static __inline void SC_SetRxRetry(SC_T *sc, uint32_t u32Count);
static __inline void SC_SET_VCC_PIN(SC_T *sc, uint32_t u32State);
static __inline void SC_SET_CLK_PIN(SC_T *sc, uint32_t u32OnOff);
static __inline void SC_SET_IO_PIN(SC_T *sc, uint32_t u32State);
static __inline void SC_SET_RST_PIN(SC_T *sc, uint32_t u32State);
# 226 "../../../../Library/StdDriver/inc\\sc.h"
static __inline void SC_SetTxRetry(SC_T *sc, uint32_t u32Count)
{
    uint32_t u32TimeOutCount = 0;

    u32TimeOutCount = (SystemCoreClock);

    while (((sc)->CTL & (0x1ul << (30))) == (0x1ul << (30)))
    {
        if (--u32TimeOutCount == 0) break;
    }


    (sc)->CTL &= ~((0x7ul << (20)) | (0x1ul << (23)));

    if ((u32Count) != 0UL)
    {
        u32TimeOutCount = (SystemCoreClock);

        while (((sc)->CTL & (0x1ul << (30))) == (0x1ul << (30)))
        {
            if (--u32TimeOutCount == 0) break;
        }

        (sc)->CTL |= (((u32Count) - 1UL) << (20)) | (0x1ul << (23));
    }
}
# 262 "../../../../Library/StdDriver/inc\\sc.h"
static __inline void SC_SetRxRetry(SC_T *sc, uint32_t u32Count)
{
    uint32_t u32TimeOutCount = 0;

    u32TimeOutCount = (SystemCoreClock);

    while (((sc)->CTL & (0x1ul << (30))) == (0x1ul << (30)))
    {
        if (--u32TimeOutCount == 0) break;
    }


    (sc)->CTL &= ~((0x7ul << (16)) | (0x1ul << (19)));

    if ((u32Count) != 0UL)
    {
        u32TimeOutCount = (SystemCoreClock);

        while (((sc)->CTL & (0x1ul << (30))) == (0x1ul << (30)))
        {
            if (--u32TimeOutCount == 0) break;
        }

        (sc)->CTL |= (((u32Count) - 1UL) << (16)) | (0x1ul << (19));
    }
}
# 299 "../../../../Library/StdDriver/inc\\sc.h"
static __inline void SC_SET_VCC_PIN(SC_T *sc, uint32_t u32State)
{
    uint32_t u32TimeOutCount = (SystemCoreClock);

    while (((sc)->PINCTL & (0x1ul << (30))) == (0x1ul << (30)))
    {
        if (--u32TimeOutCount == 0) break;
    }

    if (u32State)
    {
        (sc)->PINCTL |= (0x1ul << (0));
    }
    else
    {
        (sc)->PINCTL &= ~(0x1ul << (0));
    }
}
# 328 "../../../../Library/StdDriver/inc\\sc.h"
static __inline void SC_SET_CLK_PIN(SC_T *sc, uint32_t u32OnOff)
{
    uint32_t u32TimeOutCount = (SystemCoreClock);

    while (((sc)->PINCTL & (0x1ul << (30))) == (0x1ul << (30)))
    {
        if (--u32TimeOutCount == 0) break;
    }

    if (u32OnOff)
    {
        (sc)->PINCTL |= (0x1ul << (6));
    }
    else
    {
        (sc)->PINCTL &= ~((0x1ul << (6)));
    }
}
# 357 "../../../../Library/StdDriver/inc\\sc.h"
static __inline void SC_SET_IO_PIN(SC_T *sc, uint32_t u32State)
{
    uint32_t u32TimeOutCount = (SystemCoreClock);

    while (((sc)->PINCTL & (0x1ul << (30))) == (0x1ul << (30)))
    {
        if (--u32TimeOutCount == 0) break;
    }

    if (u32State)
    {
        (sc)->PINCTL |= (0x1ul << (9));
    }
    else
    {
        (sc)->PINCTL &= ~(0x1ul << (9));
    }

}
# 387 "../../../../Library/StdDriver/inc\\sc.h"
static __inline void SC_SET_RST_PIN(SC_T *sc, uint32_t u32State)
{
    uint32_t u32TimeOutCount = (SystemCoreClock);

    while (((sc)->PINCTL & (0x1ul << (30))) == (0x1ul << (30)))
    {
        if (--u32TimeOutCount == 0) break;
    }

    if (u32State)
    {
        (sc)->PINCTL |= (0x1ul << (1));
    }
    else
    {
        (sc)->PINCTL &= ~(0x1ul << (1));
    }

}


uint32_t SC_IsCardInserted(SC_T *sc);
void SC_ClearFIFO(SC_T *sc);
void SC_Close(SC_T *sc);
void SC_Open(SC_T *sc, uint32_t u32CardDet, uint32_t u32PWR);
void SC_ResetReader(SC_T *sc);
void SC_SetBlockGuardTime(SC_T *sc, uint32_t u32BGT);
void SC_SetCharGuardTime(SC_T *sc, uint32_t u32CGT);
void SC_StopAllTimer(SC_T *sc);
void SC_StartTimer(SC_T *sc, uint32_t u32TimerNum, uint32_t u32Mode, uint32_t u32ETUCount);
void SC_StopTimer(SC_T *sc, uint32_t u32TimerNum);
uint32_t SC_GetInterfaceClock(SC_T *sc);
# 1729 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\scu/scu.h" 1
# 52 "../../../../Library/StdDriver/inc\\scu/scu.h"
typedef enum
{
    eSCU_MASTER_ID_CPU = 0,
    eSCU_MASTER_ID_PDMA0 = 1,
    eSCU_MASTER_ID_PDMA1 = 2,
    eSCU_MASTER_ID_USBH0 = 3,
    eSCU_MASTER_ID_HSUSBH = 5,
    eSCU_MASTER_ID_HSUSBD = 6,
    eSCU_MASTER_ID_SDH0 = 7,
    eSCU_MASTER_ID_SDH1 = 8,
    eSCU_MASTER_ID_EMAC = 9,
    eSCU_MASTER_ID_CRYPTO = 10,
    eSCU_MASTER_ID_CRC = 11,
    eSCU_MASTER_ID_GDMA = 12,
    eSCU_MASTER_ID_NPU = 13,
    eSCU_MASTER_ID_LPPDMA = 14,
    eSCU_MASTER_ID_CCAP = 15,
    eSCU_MASTER_ID_SPIM0 = 16,
    eSCU_MASTER_ID_SPIM1 = 17,
} E_SCU_MASTER_ID;

typedef enum
{
    eSCU_INT_IDX_APB0 = 0,
    eSCU_INT_IDX_APB1 = 1,
    eSCU_INT_IDX_APB2 = 2,
    eSCU_INT_IDX_APB3 = 3,
    eSCU_INT_IDX_APB4 = 4,
    eSCU_INT_IDX_APB5 = 5,
    eSCU_INT_IDX_D0PPC0 = 8,
    eSCU_INT_IDX_D1PPC0 = 9,
    eSCU_INT_IDX_D1PPC1 = 10,
    eSCU_INT_IDX_D2PPC0 = 11,
    eSCU_INT_IDX_EBI = 16,
    eSCU_INT_IDX_GDMA = 32 + 0,
    eSCU_INT_IDX_PDMA0 = 32 + 1,
    eSCU_INT_IDX_PDMA1 = 32 + 2,
    eSCU_INT_IDX_USBH0 = 32 + 3,
    eSCU_INT_IDX_HSUSBH = 32 + 5,
    eSCU_INT_IDX_HSUSBD = 32 + 6,
    eSCU_INT_IDX_SDH0 = 32 + 7,
    eSCU_INT_IDX_SDH1 = 32 + 8,
    eSCU_INT_IDX_EMAC = 32 + 9,
    eSCU_INT_IDX_CRYPTO = 32 + 10,
    eSCU_INT_IDX_CRC = 32 + 11,
    eSCU_INT_IDX_LPPDMA = 32 + 12,
    eSCU_INT_IDX_CCAP = 32 + 13,
    eSCU_INT_IDX_NPUIF1 = 32 + 14,
    eSCU_INT_IDX_NPUIF0 = 32 + 15,
    eSCU_INT_IDX_SPIM0 = 32 + 16,
    eSCU_INT_IDX_SPIM1 = 32 + 17,
    eSCU_INT_IDX_SRAM0_MPC = 64 + 0,
    eSCU_INT_IDX_SRAM1_MPC = 64 + 1,
    eSCU_INT_IDX_SRAM2_MPC = 64 + 2,
    eSCU_INT_IDX_SRAM3_MPC = 64 + 3,
    eSCU_INT_IDX_LPSRAM_MPC = 64 + 4,
    eSCU_INT_IDX_SPIM0_MPC = 64 + 5,
    eSCU_INT_IDX_SPIM1_MPC = 64 + 6,
} E_SCU_INT_IDX;




typedef enum
{

    eSCU_PERI_IDX_NPU = 3,


    eSCU_PERI_IDX_SPIM0 = 64 + 2,
    eSCU_PERI_IDX_SPIM1 = 64 + 3,


    eSCU_PERI_IDX_PDMA0 = 256 + 0,
    eSCU_PERI_IDX_PDMA1 = 256 + 1,
    eSCU_PERI_IDX_USBH0 = 256 + 2,
    eSCU_PERI_IDX_HSUSBH = 256 + 4,
    eSCU_PERI_IDX_HSUSBD = 256 + 5,
    eSCU_PERI_IDX_SDH0 = 256 + 6,
    eSCU_PERI_IDX_SDH1 = 256 + 7,
    eSCU_PERI_IDX_EMAC0 = 256 + 8,
    eSCU_PERI_IDX_CRYPTO = 256 + 10,
    eSCU_PERI_IDX_CRC = 256 + 11,


    eSCU_PERI_IDX_KDF = 288 + 1,
    eSCU_PERI_IDX_CANFD0 = 288 + 2,
    eSCU_PERI_IDX_CANFD1 = 288 + 4,
    eSCU_PERI_IDX_ETMC = 288 + 6,
    eSCU_PERI_IDX_SWDH = 288 + 7,
    eSCU_PERI_IDX_SWODEC = 288 + 8,
    eSCU_PERI_IDX_EBI = 288 + 16,


    eSCU_PERI_IDX_WWDT0 = 320 + 0,
    eSCU_PERI_IDX_EADC0 = 320 + 1,
    eSCU_PERI_IDX_EPWM0 = 320 + 2,
    eSCU_PERI_IDX_BPWM0 = 320 + 3,
    eSCU_PERI_IDX_EQEI0 = 320 + 4,
    eSCU_PERI_IDX_EQEI2 = 320 + 5,
    eSCU_PERI_IDX_ECAP0 = 320 + 6,
    eSCU_PERI_IDX_ECAP2 = 320 + 7,
    eSCU_PERI_IDX_I2C0 = 320 + 8,
    eSCU_PERI_IDX_I2C2 = 320 + 9,
    eSCU_PERI_IDX_QSPI0 = 320 + 10,
    eSCU_PERI_IDX_SPI0 = 320 + 11,
    eSCU_PERI_IDX_SPI2 = 320 + 12,
    eSCU_PERI_IDX_UART0 = 320 + 13,
    eSCU_PERI_IDX_UART2 = 320 + 14,
    eSCU_PERI_IDX_UART4 = 320 + 15,
    eSCU_PERI_IDX_UART6 = 320 + 16,
    eSCU_PERI_IDX_UART8 = 320 + 17,
    eSCU_PERI_IDX_USCI0 = 320 + 18,
    eSCU_PERI_IDX_SC0 = 320 + 19,
    eSCU_PERI_IDX_SC2 = 320 + 20,
    eSCU_PERI_IDX_PSIO = 320 + 21,
    eSCU_PERI_IDX_TMR01 = 320 + 22,
    eSCU_PERI_IDX_DAC01 = 320 + 23,
    eSCU_PERI_IDX_HSOTG = 320 + 25,
    eSCU_PERI_IDX_I2S0 = 320 + 26,
    eSCU_PERI_IDX_ACMP01 = 320 + 27,
    eSCU_PERI_IDX_USBD = 320 + 28,


    eSCU_PERI_IDX_WWDT1 = 384 + 0,
    eSCU_PERI_IDX_EADC1 = 384 + 1,
    eSCU_PERI_IDX_EPWM1 = 384 + 2,
    eSCU_PERI_IDX_BPWM1 = 384 + 3,
    eSCU_PERI_IDX_EQEI1 = 384 + 4,
    eSCU_PERI_IDX_EQEI3 = 384 + 5,
    eSCU_PERI_IDX_ECAP1 = 384 + 6,
    eSCU_PERI_IDX_ECAP3 = 384 + 7,
    eSCU_PERI_IDX_I2C1 = 384 + 8,
    eSCU_PERI_IDX_I2C3 = 384 + 9,
    eSCU_PERI_IDX_QSPI1 = 384 + 10,
    eSCU_PERI_IDX_SPI1 = 384 + 11,
    eSCU_PERI_IDX_SPI3 = 384 + 12,
    eSCU_PERI_IDX_UART1 = 384 + 13,
    eSCU_PERI_IDX_UART3 = 384 + 14,
    eSCU_PERI_IDX_UART5 = 384 + 15,
    eSCU_PERI_IDX_UART7 = 384 + 16,
    eSCU_PERI_IDX_UART9 = 384 + 17,
    eSCU_PERI_IDX_SC1 = 384 + 18,
    eSCU_PERI_IDX_OTG = 384 + 19,
    eSCU_PERI_IDX_KPI = 384 + 20,
    eSCU_PERI_IDX_TMR23 = 384 + 21,
    eSCU_PERI_IDX_TRNG = 384 + 22,
    eSCU_PERI_IDX_RTC = 384 + 23,
    eSCU_PERI_IDX_I2S1 = 384 + 24,
    eSCU_PERI_IDX_ACMP23 = 384 + 25,
    eSCU_PERI_IDX_I3C0 = 384 + 26,
    eSCU_PERI_IDX_UTCPD = 384 + 27,


    eSCU_PERI_IDX_LPPDMA = 512 + 0,
    eSCU_PERI_IDX_CCAP = 512 + 1,
    eSCU_PERI_IDX_LPGPIO = 512 + 3,


    eSCU_PERI_IDX_LPTMR01 = 576 + 0,
    eSCU_PERI_IDX_TTMR01 = 576 + 1,
    eSCU_PERI_IDX_LPADC0 = 576 + 2,
    eSCU_PERI_IDX_LPI2C0 = 576 + 3,
    eSCU_PERI_IDX_LPSPI0 = 576 + 4,
    eSCU_PERI_IDX_DMIC0 = 576 + 5,
    eSCU_PERI_IDX_LPUART0 = 576 + 6,
    eSCU_PERI_IDX_AWF = 576 + 9,
} E_SCU_PERI_IDX;
# 450 "../../../../Library/StdDriver/inc\\scu/scu.h"
static __inline void SCU_NSMConfig(uint32_t u32Ticks, uint32_t u32Prescale);
static __inline void SCU_TimerConfig(uint32_t u32Ticks, uint32_t u32Prescale);
# 470 "../../../../Library/StdDriver/inc\\scu/scu.h"
static __inline void SCU_NSMConfig(uint32_t u32Ticks, uint32_t u32Prescale)
{

    ((SCU_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02000UL))->NSMLOAD = u32Ticks;
    ((SCU_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02000UL))->NSMVAL = 0ul;
    ((SCU_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02000UL))->NSMCTL = (0x1ul << (9)) | (0x1ul << (8)) | (u32Prescale & 0xFFul);
}
# 491 "../../../../Library/StdDriver/inc\\scu/scu.h"
static __inline void SCU_TimerConfig(uint32_t u32Ticks, uint32_t u32Prescale)
{

    ((SCU_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02000UL))->NSMLOAD = u32Ticks;
    ((SCU_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02000UL))->NSMVAL = 0ul;
    ((SCU_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02000UL))->NSMCTL = (0x1ul << (9)) | (0x1ul << (8)) | (0x1ul << (10)) | (u32Prescale & 0xFFul);
}
# 1730 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\scu/dpm.h" 1
# 30 "../../../../Library/StdDriver/inc\\scu/dpm.h"
typedef enum
{
    eDPM_STS_DEFAULT = 0,
    eDPM_STS_CLOSE = 1,
    eDPM_STS_LOCKED = 2,
    eDPM_STS_OPEN = 5
} E_DPM_STS;
# 61 "../../../../Library/StdDriver/inc\\scu/dpm.h"
extern int32_t g_DPM_i32ErrCode;
# 170 "../../../../Library/StdDriver/inc\\scu/dpm.h"
void DPM_SetDebugDisable(uint32_t u32dpm);
void DPM_SetDebugLock(uint32_t u32dpm);
int32_t DPM_GetDebugDisable(uint32_t u32dpm);
int32_t DPM_GetDebugLock(uint32_t u32dpm);
int32_t DPM_SetPasswordUpdate(uint32_t u32dpm, uint32_t au32Pwd[]);
int32_t DPM_SetPasswordCompare(uint32_t u32dpm, uint32_t au32Pwd[]);
int32_t DPM_GetPasswordErrorFlag(uint32_t u32dpm);
int32_t DPM_GetIntFlag(void);
void DPM_ClearPasswordErrorFlag(uint32_t u32dpm);
void DPM_EnableDebuggerWriteAccess(uint32_t u32dpm);
void DPM_DisableDebuggerWriteAccess(uint32_t u32dpm);
# 1731 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\scu/plm.h" 1
# 30 "../../../../Library/StdDriver/inc\\scu/plm.h"
typedef enum
{
    PLM_VENDOR = 0,
    PLM_OEM = 1,
    PLM_DEPLOYED = 3,
    PLM_RMA = 7
} PLM_STAGE_T;
# 70 "../../../../Library/StdDriver/inc\\scu/plm.h"
static __inline int32_t PLM_SetStage(PLM_STAGE_T eStage)
{


    if (((((PLM_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02700UL))->STS & (0x7ul << (0))) >> (0)) == eStage)
        return 0;


    if (((PLM_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02700UL))->STS & (0x1ul << (8)))
        return -1;

    ((PLM_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02700UL))->CTL = (0x475AUL << (16)) | (eStage);


    if (((PLM_T *) ((((uint32_t) 0x40000000UL) + 0x00400000UL) + 0x02700UL))->STS & (0x1ul << (8)))
        return -1;

    return 0;
}
# 1732 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\scu/mpc_sie_reg_map.h" 1
# 33 "../../../../Library/StdDriver/inc\\scu/mpc_sie_reg_map.h"
struct mpc_sie_reg_map_t
{
    volatile uint32_t ctrl;
    volatile uint32_t reserved[3];
    volatile uint32_t blk_max;
    volatile uint32_t blk_cfg;
    volatile uint32_t blk_idx;


    volatile uint32_t blk_lutn;


    volatile uint32_t int_stat;
    volatile uint32_t int_clear;
    volatile uint32_t int_en;
    volatile uint32_t int_info1;
    volatile uint32_t int_info2;
    volatile uint32_t int_set;
    volatile uint32_t reserved2[998];
    volatile uint32_t pidr4;
    volatile uint32_t pidr5;
    volatile uint32_t pidr6;
    volatile uint32_t pidr7;
    volatile uint32_t pidr0;
    volatile uint32_t pidr1;
    volatile uint32_t pidr2;
    volatile uint32_t pidr3;
    volatile uint32_t cidr0;
    volatile uint32_t cidr1;
    volatile uint32_t cidr2;
    volatile uint32_t cidr3;
};
# 1733 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h" 1
# 52 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t
{
    MPC_SIE_ERR_NONE,
    MPC_SIE_INVALID_ARG,
    MPC_SIE_NOT_INIT,
    MPC_SIE_ERR_NOT_IN_RANGE,

    MPC_SIE_ERR_NOT_ALIGNED,


    MPC_SIE_ERR_INVALID_RANGE,





    MPC_SIE_ERR_RANGE_SEC_ATTR_NON_COMPATIBLE,



    MPC_SIE_UNSUPPORTED_HARDWARE_VERSION,


    MPC_SIE_ERR_GATING_NOT_PRESENT
};


enum mpc_sie_sec_attr_t
{
    MPC_SIE_SEC_ATTR_SECURE,
    MPC_SIE_SEC_ATTR_NONSECURE,



    MPC_SIE_SEC_ATTR_MIXED,
};


enum mpc_sie_sec_resp_t
{
    MPC_SIE_RESP_RAZ_WI,
    MPC_SIE_RESP_BUS_ERROR,
    MPC_SIE_RESP_WAIT_GATING_DISABLED
};


struct mpc_sie_memory_range_t
{
    const uint32_t base;
    const uint32_t limit;
    const uint32_t range_offset;




    const enum mpc_sie_sec_attr_t attr;
# 117 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
};


struct mpc_sie_dev_cfg_t
{
    const uint32_t base;

    const struct mpc_sie_memory_range_t **range_list;
    uint8_t nbr_of_ranges;
};


struct mpc_sie_dev_data_t
{
    _Bool is_initialized;


    uint32_t sie_version;
};


struct mpc_sie_dev_t
{
    const struct mpc_sie_dev_cfg_t *const cfg;
    struct mpc_sie_dev_data_t *const data;
};
# 153 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_init(struct mpc_sie_dev_t *dev);
# 166 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_get_block_size(struct mpc_sie_dev_t *dev,
                                            uint32_t *blk_size);
# 193 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_config_region(struct mpc_sie_dev_t *dev,
                                           const uint32_t base,
                                           const uint32_t limit,
                                           enum mpc_sie_sec_attr_t attr);
# 211 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_get_region_config(struct mpc_sie_dev_t *dev,
                                               uint32_t base,
                                               uint32_t limit,
                                               enum mpc_sie_sec_attr_t *attr);
# 226 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_get_ctrl(struct mpc_sie_dev_t *dev,
                                      uint32_t *ctrl_val);
# 239 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_set_ctrl(struct mpc_sie_dev_t *dev,
                                      uint32_t mpc_ctrl);
# 252 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_get_sec_resp(struct mpc_sie_dev_t *dev,
                                          enum mpc_sie_sec_resp_t *sec_rep);
# 263 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_set_sec_resp(struct mpc_sie_dev_t *dev,
                                          enum mpc_sie_sec_resp_t sec_rep);
# 275 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_irq_enable(struct mpc_sie_dev_t *dev);
# 284 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
void mpc_sie_irq_disable(struct mpc_sie_dev_t *dev);
# 293 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
void mpc_sie_clear_irq(struct mpc_sie_dev_t *dev);
# 304 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
uint32_t mpc_sie_irq_state(struct mpc_sie_dev_t *dev);
# 315 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_lock_down(struct mpc_sie_dev_t *dev);
# 327 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
enum mpc_sie_error_t mpc_sie_is_gating_present(struct mpc_sie_dev_t *dev,
                                               _Bool *gating_present);
# 339 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
uint32_t get_sie_version(struct mpc_sie_dev_t *dev);
# 350 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
_Bool mpc_sie_get_gate_ack(struct mpc_sie_dev_t *dev);
# 359 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
void mpc_sie_request_gating(struct mpc_sie_dev_t *dev);
# 368 "../../../../Library/StdDriver/inc\\scu/mpc_sie_drv.h"
void mpc_sie_release_gating(struct mpc_sie_dev_t *dev);
# 1734 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\scuart.h" 1
# 336 "../../../../Library/StdDriver/inc\\scuart.h"
void SCUART_Close(SC_T *sc);
uint32_t SCUART_Open(SC_T *sc, uint32_t u32Baudrate);
uint32_t SCUART_Read(SC_T *sc, uint8_t pu8RxBuf[], uint32_t u32ReadBytes);
uint32_t SCUART_SetLineConfig(SC_T *sc, uint32_t u32Baudrate, uint32_t u32DataWidth, uint32_t u32Parity, uint32_t u32StopBits);
void SCUART_SetTimeoutCnt(SC_T *sc, uint32_t u32TOC);
uint32_t SCUART_Write(SC_T *sc, uint8_t pu8TxBuf[], uint32_t u32WriteBytes);
# 1735 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\sdh.h" 1
# 10 "../../../../Library/StdDriver/inc\\sdh.h"
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
# 11 "../../../../Library/StdDriver/inc\\sdh.h" 2
# 86 "../../../../Library/StdDriver/inc\\sdh.h"
typedef struct SDH_info_t
{
    unsigned char IsCardInsert;
    unsigned char R3Flag;
    unsigned char R7Flag;
    unsigned char volatile DataReadyFlag;
    unsigned int CardType;
    unsigned int RCA;
    unsigned int totalSectorN;
    unsigned int diskSize;
    int sectorSize;
    unsigned char *dmabuf;
    int32_t i32ErrCode;
} SDH_INFO_T;




extern SDH_INFO_T SD0, SD1;
extern int32_t g_SDH_i32ErrCode;
# 209 "../../../../Library/StdDriver/inc\\sdh.h"
void *SDH_GetSDH0Buffer(void);
void *SDH_GetSDH1Buffer(void);
void *SDH_GetSDInfoMsg(SDH_T *sdh);

void SDH_Open(SDH_T *sdh, uint32_t u32CardDetSrc);
void SDH_Close(SDH_T *sdh);
uint32_t SDH_Probe(SDH_T *sdh);
uint32_t SDH_Read(SDH_T *sdh, uint8_t *pu8BufAddr, uint32_t u32StartSec, uint32_t u32SecCount);
uint32_t SDH_Write(SDH_T *sdh, uint8_t *pu8BufAddr, uint32_t u32StartSec, uint32_t u32SecCount);

uint32_t SDH_CardDetection(SDH_T *sdh);
void SDH_Get_SD_info(SDH_T *sdh);
void SDH_Set_clock(SDH_T *sdh, uint32_t sd_clock_khz);

int32_t SDH_CheckRB(SDH_T *sdh);
uint32_t SDH_SDCmdAndRsp(SDH_T *sdh, uint32_t ucCmd, uint32_t uArg, uint32_t ntickCount);
uint32_t SDH_SDCmdAndRsp2(SDH_T *sdh, uint32_t ucCmd, uint32_t uArg, uint32_t puR2ptr[]);
uint32_t SDH_SDCommand(SDH_T *sdh, uint32_t ucCmd, uint32_t uArg);
uint32_t SDH_SelectCardType(SDH_T *sdh);

int32_t SDH_Open_Disk(SDH_T *sdh, uint32_t u32CardDetSrc);
void SDH_Close_Disk(SDH_T *sdh);
# 1736 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\spi.h" 1
# 371 "../../../../Library/StdDriver/inc\\spi.h"
static __inline void SPII2S_ENABLE_TX_ZCD(SPI_T *i2s, uint32_t u32ChMask);
static __inline void SPII2S_DISABLE_TX_ZCD(SPI_T *i2s, uint32_t u32ChMask);
static __inline void SPII2S_SET_MONO_RX_CHANNEL(SPI_T *i2s, uint32_t u32Ch);
# 383 "../../../../Library/StdDriver/inc\\spi.h"
static __inline void SPII2S_ENABLE_TX_ZCD(SPI_T *i2s, uint32_t u32ChMask)
{
    if (u32ChMask == (0U))
    {
        i2s->I2SCTL |= (0x1ul << (16));
    }
    else
    {
        i2s->I2SCTL |= (0x1ul << (17));
    }
}
# 403 "../../../../Library/StdDriver/inc\\spi.h"
static __inline void SPII2S_DISABLE_TX_ZCD(SPI_T *i2s, uint32_t u32ChMask)
{
    if (u32ChMask == (0U))
    {
        i2s->I2SCTL &= ~(0x1ul << (16));
    }
    else
    {
        i2s->I2SCTL &= ~(0x1ul << (17));
    }
}
# 518 "../../../../Library/StdDriver/inc\\spi.h"
static __inline void SPII2S_SET_MONO_RX_CHANNEL(SPI_T *i2s, uint32_t u32Ch)
{
    u32Ch == (0x1ul << (23)) ?
    (i2s->I2SCTL |= (0x1ul << (23))) :
    (i2s->I2SCTL &= ~(0x1ul << (23)));
}
# 585 "../../../../Library/StdDriver/inc\\spi.h"
uint32_t SPI_Open(SPI_T *spi, uint32_t u32MasterSlave, uint32_t u32SPIMode, uint32_t u32DataWidth, uint32_t u32BusClock);
void SPI_Close(SPI_T *spi);
void SPI_ClearRxFIFO(SPI_T *spi);
void SPI_ClearTxFIFO(SPI_T *spi);
void SPI_DisableAutoSS(SPI_T *spi);
void SPI_EnableAutoSS(SPI_T *spi, uint32_t u32SSPinMask, uint32_t u32ActiveLevel);
uint32_t SPI_SetBusClock(SPI_T *spi, uint32_t u32BusClock);
void SPI_SetFIFO(SPI_T *spi, uint32_t u32TxThreshold, uint32_t u32RxThreshold);
uint32_t SPI_GetBusClock(SPI_T *spi);
void SPI_EnableInt(SPI_T *spi, uint32_t u32Mask);
void SPI_DisableInt(SPI_T *spi, uint32_t u32Mask);
uint32_t SPI_GetIntFlag(SPI_T *spi, uint32_t u32Mask);
void SPI_ClearIntFlag(SPI_T *spi, uint32_t u32Mask);
uint32_t SPI_GetStatus(SPI_T *spi, uint32_t u32Mask);
uint32_t SPI_GetStatus2(SPI_T *spi, uint32_t u32Mask);


uint32_t SPII2S_Open(SPI_T *i2s, uint32_t u32MasterSlave, uint32_t u32SampleRate, uint32_t u32WordWidth, uint32_t u32Channels, uint32_t u32DataFormat);
void SPII2S_Close(SPI_T *i2s);
void SPII2S_EnableInt(SPI_T *i2s, uint32_t u32Mask);
void SPII2S_DisableInt(SPI_T *i2s, uint32_t u32Mask);
uint32_t SPII2S_EnableMCLK(SPI_T *i2s, uint32_t u32BusClock);
void SPII2S_DisableMCLK(SPI_T *i2s);
void SPII2S_SetFIFO(SPI_T *i2s, uint32_t u32TxThreshold, uint32_t u32RxThreshold);
# 1737 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\spim.h" 1
# 202 "../../../../Library/StdDriver/inc\\spim.h"
typedef enum
{
    MFGID_UNKNOW = 0x00U,
    MFGID_SPANSION = 0x01U,
    MFGID_EON = 0x1CU,
    MFGID_ISSI = 0x7FU,
    MFGID_MXIC = 0xC2U,
    MFGID_WINBOND = 0xEFU,
    MFGID_MICRON = 0x2CU,
} E_MFGID;
# 1657 "../../../../Library/StdDriver/inc\\spim.h"
static __inline void SPIM_DISABLE_CIPHER(SPIM_T *spim);
static __inline void SPIM_ENABLE_CIPHER(SPIM_T *spim);

static __inline void SPIM_DISABLE_CACHE(SPIM_T *spim);
static __inline void SPIM_ENABLE_CACHE(SPIM_T *spim);

static __inline uint32_t SPIM_GetDMMAddress(SPIM_T *spim);







static __inline void SPIM_ENABLE_CIPHER(SPIM_T *spim)
{
    spim->CTL0 &= ~((0x1ul << (0)) | (0x1ul << (2)));

    (spim->DMMCTL = (spim->DMMCTL & ~(0xfful << (16))) | (((0x12)&0x1FUL) << (16)));
}







static __inline void SPIM_DISABLE_CIPHER(SPIM_T *spim)
{
    spim->CTL0 |= ((0x1ul << (2)) | (0x1ul << (0)));

    (spim->DMMCTL = (spim->DMMCTL & ~(0xfful << (16))) | (((0x08)&0x1FUL) << (16)));
}







static __inline void SPIM_DISABLE_CACHE(SPIM_T *spim)
{





    if (((spim->CTL0 & (0x1ul << (0))) >> (0)) != (0x00UL))
    {
        (spim->DMMCTL = (spim->DMMCTL & ~(0xfful << (16))) | (((0x1)&0x1FUL) << (16)));
    }
}







static __inline void SPIM_ENABLE_CACHE(SPIM_T *spim)
{
# 1728 "../../../../Library/StdDriver/inc\\spim.h"
}






static __inline uint32_t SPIM_GetDMMAddress(SPIM_T *spim)
{
    uint32_t u32DMMAddr = 0;

    if (spim == ((SPIM_T *) ((((uint32_t) 0x40000000UL) + 0x00040000UL) + 0x02000UL)))
    {
        u32DMMAddr = (0x80000000UL);
    }
    else if (spim == ((SPIM_T *) ((((uint32_t) 0x40000000UL) + 0x00040000UL) + 0x03000UL)))
    {
        u32DMMAddr = (0x82000000UL);
    }

    return u32DMMAddr;
}




typedef struct
{
    uint32_t u32CMDCode;

    uint32_t u32CMDPhase;
    uint32_t u32CMDWidth;
    uint32_t u32CMDDTR;

    uint32_t u32AddrPhase;
    uint32_t u32AddrWidth;
    uint32_t u32AddrDTR;

    uint32_t u32DataPhase;
    uint32_t u32ByteOrder;
    uint32_t u32DataDTR;
    uint32_t u32RDQS;

    uint32_t u32DcNum;



    uint32_t u32ContRdEn;
    uint32_t u32RdModePhase;
    uint32_t u32RdModeWidth;
    uint32_t u32RdModeDTR;

} SPIM_PHASE_T;
# 1791 "../../../../Library/StdDriver/inc\\spim.h"
int32_t SPIM_SetDLLDelayNum(SPIM_T *spim, uint32_t u32DelayNum);




int32_t SPIM_InitFlash(SPIM_T *spim, int clrWP);
uint32_t SPIM_GetSClkFreq(SPIM_T *spim);
void SPIM_ReadJedecId(SPIM_T *spim, uint8_t *idBuf, uint32_t u32NRx, uint32_t u32NBit, uint32_t u32DcNum);
int32_t SPIM_Enable_4Bytes_Mode(SPIM_T *spim, int isEn, uint32_t u32NBit);
int32_t SPIM_Is4ByteModeEnable(SPIM_T *spim, uint32_t u32NBit);
int32_t SPIM_SetWrapAroundEnable(SPIM_T *spim, int isEn);
void SPIM_SetWriteEnable(SPIM_T *spim, int isEn, uint32_t u32NBit);

void SPIM_ReadStatusRegister(SPIM_T *spim, uint8_t dataBuf[], uint32_t u32NRx, uint32_t u32NBit);
void SPIM_ReadConfigRegister(SPIM_T *spim, uint8_t u8RdCMD, uint32_t u32Addr, uint8_t *pu8DataBuf, uint32_t u32NRx, uint32_t u32NBit);
void SPIM_WriteConfigRegister(SPIM_T *spim, uint8_t u8CMD, uint32_t u32Addr, uint8_t u8Data, uint32_t u32NBit);

void SPIM_ChipErase(SPIM_T *spim, uint32_t u32NBit, int isSync);
void SPIM_EraseBlock(SPIM_T *spim, uint32_t u32Addr, int is4ByteAddr, uint8_t u8ErsCmd, uint32_t u32NBit, int isSync);

void SPIM_IO_Write(SPIM_T *spim, uint32_t u32Addr, int is4ByteAddr, uint32_t u32NTx,
                   uint8_t *pu8TxBuf, uint8_t wrCmd, uint32_t u32NBitCmd,
                   uint32_t u32NBitAddr, uint32_t u32NBitDat);
void SPIM_IO_Read(SPIM_T *spim, uint32_t u32Addr, int is4ByteAddr, uint32_t u32RdSize,
                  uint8_t *pu8RxBuf, uint8_t rdCmd, uint32_t u32NBitCmd,
                  uint32_t u32NBitAddr, uint32_t u32NBitDat, int u32NDummy);

void SPIM_DMA_Write(SPIM_T *spim, uint32_t u32Addr, int is4ByteAddr, uint32_t u32NTx, uint8_t *pu8TxBuf, uint32_t wrCmd);
int32_t SPIM_DMA_Read(SPIM_T *spim, uint32_t u32Addr, int is4ByteAddr, uint32_t u32NRx, uint8_t *pu8RxBuf, uint32_t u32RdCmd, int isSync);

void SPIM_EnterDirectMapMode(SPIM_T *spim, int is4ByteAddr, uint32_t u32RdCmd, uint32_t u32IdleIntvl);
void SPIM_ExitDirectMapMode(SPIM_T *spim);

int32_t SPIM_SetContReadDisable(SPIM_T *spim);

void SPIM_SetQuadEnable(SPIM_T *spim, int isEn, uint32_t u32NBit);

void SPIM_WinbondUnlock(SPIM_T *spim, uint32_t u32NBit);


void SPIM_DMADMM_SetRWDQS(SPIM_T *spim, uint32_t u32OPMode, uint32_t u32RdDQS);


void SPIM_DMADMM_InitPhase(SPIM_T *spim, SPIM_PHASE_T *psPhaseTable, uint32_t u32OPMode);


uint32_t SPIM_PhaseModeToNBit(uint32_t u32Phase);


void SPIM_IO_SendCMDPhase(SPIM_T *spim, uint32_t u32OPMode, uint32_t u32OpCMD, uint32_t u32CMDPhase, uint32_t u32DTREn);

void SPIM_IO_SendAddrPhase(SPIM_T *spim, uint32_t u32Is4ByteAddr, uint32_t u32Addr, uint32_t u32AddrPhase, uint32_t u32DTREn);

int32_t SPIM_IO_SendDummyCycle(SPIM_T *spim, int u32NDummy);

void SPIM_IO_SendDataPhase(SPIM_T *spim, uint32_t u32OPMode, uint8_t *pu8TRxBuf, uint32_t u32TRxSize, uint32_t u32DataPhase, uint32_t u32DTREn, uint32_t u32RdDQS);


void SPIM_IO_WritePhase(SPIM_T *spim, SPIM_PHASE_T *psPhaseTable, uint32_t u32Addr, uint8_t *pu8RxBuf, uint32_t u32WrSize);

void SPIM_IO_ReadPhase(SPIM_T *spim, SPIM_PHASE_T *psPhaseTable, uint32_t u32Addr, uint8_t *pu8RxBuf, uint32_t u32RdSize);
# 1738 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\spim_hyper.h" 1
# 654 "../../../../Library/StdDriver/inc\\spim_hyper.h"
static __inline void SPIM_HYPER_DISABLE_CIPHER(SPIM_T *spim);
static __inline void SPIM_HYPER_ENABLE_CIPHER(SPIM_T *spim);

static __inline void SPIM_HYPER_DISABLE_CACHE(SPIM_T *spim);
static __inline void SPIM_HYPER_ENABLE_CACHE(SPIM_T *spim);
static __inline uint32_t SPIM_HYPER_GetDMMAddress(SPIM_T *spim);







static __inline void SPIM_HYPER_DISABLE_CIPHER(SPIM_T *spim)
{
    spim->CTL0 |= ((0x1ul << (2)) | (0x1ul << (0)));


    (spim->DMMCTL = (spim->DMMCTL & ~(0xfful << (16))) | (((0x08) & 0x1FUL) << (16)));
}







static __inline void SPIM_HYPER_ENABLE_CIPHER(SPIM_T *spim)
{
    spim->CTL0 &= ~((0x1ul << (0)) | (0x1ul << (2)));


    (spim->DMMCTL = (spim->DMMCTL & ~(0xfful << (16))) | (((0x12) & 0x1FUL) << (16)));
}







static __inline void SPIM_HYPER_DISABLE_CACHE(SPIM_T *spim)
{





    if (((spim->CTL0 & (0x1ul << (0))) >> (0)) != (0x00UL))
    {
        (spim->DMMCTL = (spim->DMMCTL & ~(0xfful << (16))) | (((0x1) & 0x1FUL) << (16)));
    }
}







static __inline void SPIM_HYPER_ENABLE_CACHE(SPIM_T *spim)
{
# 726 "../../../../Library/StdDriver/inc\\spim_hyper.h"
}






static __inline uint32_t SPIM_HYPER_GetDMMAddress(SPIM_T *spim)
{
    uint32_t u32DMMAddr = 0;

    if (spim == ((SPIM_T *) ((((uint32_t) 0x40000000UL) + 0x00040000UL) + 0x02000UL)))
    {
        u32DMMAddr = (0x80000000UL);
    }
    else if (spim == ((SPIM_T *) ((((uint32_t) 0x40000000UL) + 0x00040000UL) + 0x03000UL)))
    {
        u32DMMAddr = (0x82000000UL);
    }

    return u32DMMAddr;
}


int32_t SPIM_HYPER_SetDLLDelayNum(SPIM_T *spim, uint32_t u32DelayNum);


int32_t SPIM_HYPER_ExitHSAndDPD(SPIM_T *spim);
int32_t SPIM_HYPER_ReadHyperRAMReg(SPIM_T *spim, uint32_t u32Addr);
int32_t SPIM_HYPER_WriteHyperRAMReg(SPIM_T *spim, uint32_t u32Addr, uint32_t u32Value);


void SPIM_HYPER_Init(SPIM_T *spim, uint32_t u32Div);
int32_t SPIM_HYPER_Reset(SPIM_T *spim);
int16_t SPIM_HYPER_Read1Word(SPIM_T *spim, uint32_t u32Addr);
int32_t SPIM_HYPER_Read2Word(SPIM_T *spim, uint32_t u32Addr);
int32_t SPIM_HYPER_Write1Byte(SPIM_T *spim, uint32_t u32Addr, uint8_t u8Data);
int32_t SPIM_HYPER_Write2Byte(SPIM_T *spim, uint32_t u32Addr, uint16_t u16Data);
int32_t SPIM_HYPER_Write3Byte(SPIM_T *spim, uint32_t u32Addr, uint32_t u32Data);
int32_t SPIM_HYPER_Write4Byte(SPIM_T *spim, uint32_t u32Addr, uint32_t u32Data);

int32_t SPIM_HYPER_DMAWrite(SPIM_T *spim, uint32_t u32Addr, uint8_t *pu8WrBuf, uint32_t u32NTx);
int32_t SPIM_HYPER_DMARead(SPIM_T *spim, uint32_t u32Addr, uint8_t *pu8RdBuf, uint32_t u32NRx);

void SPIM_HYPER_EnterDirectMapMode(SPIM_T *spim);
void SPIM_HYPER_ExitDirectMapMode(SPIM_T *spim);
int32_t SPIM_HYPER_IsDMMDone(SPIM_T *spim);
# 1739 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\sys.h" 1
# 4454 "../../../../Library/StdDriver/inc\\sys.h"
static __inline void SYS_UnlockReg(void);
static __inline void SYS_LockReg(void);
static __inline void SYS_ClearResetSrc(uint32_t u32Src);
static __inline uint32_t SYS_GetBODStatus(void);
static __inline uint32_t SYS_GetResetSrc(void);
static __inline uint32_t SYS_IsRegLocked(void);
static __inline uint32_t SYS_ReadPDID(void);
static __inline void SYS_ResetChip(void);
static __inline void SYS_ResetCPU(void);
static __inline void SYS_SetVRef(uint32_t u32VRefCTL);
# 4472 "../../../../Library/StdDriver/inc\\sys.h"
static __inline void SYS_UnlockReg(void)
{
    uint32_t u32TimeOutCount = SystemCoreClock;

    do
    {
        ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->REGLCTL = 0x59UL;
        ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->REGLCTL = 0x16UL;
        ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->REGLCTL = 0x88UL;

        if (--u32TimeOutCount == 0) break;
    } while (((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->REGLCTL == 0UL);
}
# 4493 "../../../../Library/StdDriver/inc\\sys.h"
static __inline void SYS_LockReg(void)
{
    ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->REGLCTL = 0UL;
}
# 4512 "../../../../Library/StdDriver/inc\\sys.h"
void SYS_ClearResetSrc(uint32_t u32Src)
{
    ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->RSTSTS = u32Src;
}
# 4524 "../../../../Library/StdDriver/inc\\sys.h"
static __inline uint32_t SYS_GetBODStatus(void)
{
    return ((((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->BODCTL & (0x1ul << (6))) >> (6));
}







static __inline uint32_t SYS_GetResetSrc(void)
{
    return (((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->RSTSTS);
}
# 4547 "../../../../Library/StdDriver/inc\\sys.h"
static __inline uint32_t SYS_IsRegLocked(void)
{
    return ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->REGLCTL & 1UL ? 0UL : 1UL;
}







static __inline uint32_t SYS_ReadPDID(void)
{
    return ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->PDID;
}
# 4570 "../../../../Library/StdDriver/inc\\sys.h"
static __inline void SYS_ResetChip(void)
{
    ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->RSTCTL |= (0x1ul << (0));
}
# 4582 "../../../../Library/StdDriver/inc\\sys.h"
static __inline void SYS_ResetCPU(void)
{
    ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->RSTCTL |= (0x1ul << (7));

    ((SCB_Type *) ((0xE000E000UL) + 0x0D00UL) )->SCR &= ~(1UL << 2U);

    __builtin_arm_wfi();
}
# 4604 "../../../../Library/StdDriver/inc\\sys.h"
static __inline void SYS_SetVRef(uint32_t u32VRefCTL)
{

    ((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->VREFCTL = (((SYS_T *) ((((uint32_t) 0x40000000UL) + 0x00000000UL) + 0x00000UL))->VREFCTL & (~(0x1ful << (0)))) | (u32VRefCTL);
}

void SYS_ResetModule(uint32_t u32ModuleIndex);
int32_t SYS_EnableBOD(int32_t i32Mode, uint32_t u32BODLevel);
int32_t SYS_DisableBOD(void);
# 1740 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\timer.h" 1
# 225 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_Start(TIMER_T *timer);
static __inline void TIMER_Stop(TIMER_T *timer);
static __inline void TIMER_EnableWakeup(TIMER_T *timer);
static __inline void TIMER_DisableWakeup(TIMER_T *timer);
static __inline void TIMER_StartCapture(TIMER_T *timer);
static __inline void TIMER_StopCapture(TIMER_T *timer);
static __inline void TIMER_EnableCaptureDebounce(TIMER_T *timer);
static __inline void TIMER_DisableCaptureDebounce(TIMER_T *timer);
static __inline void TIMER_EnableEventCounterDebounce(TIMER_T *timer);
static __inline void TIMER_DisableEventCounterDebounce(TIMER_T *timer);
static __inline void TIMER_EnableInt(TIMER_T *timer);
static __inline void TIMER_DisableInt(TIMER_T *timer);
static __inline void TIMER_EnableCaptureInt(TIMER_T *timer);
static __inline void TIMER_DisableCaptureInt(TIMER_T *timer);
static __inline uint32_t TIMER_GetIntFlag(TIMER_T *timer);
static __inline void TIMER_ClearIntFlag(TIMER_T *timer);
static __inline uint32_t TIMER_GetCaptureIntFlag(TIMER_T *timer);
static __inline uint32_t TIMER_GetCaptureIntFlagOV(TIMER_T *timer);
static __inline void TIMER_ClearCaptureIntFlag(TIMER_T *timer);
static __inline uint32_t TIMER_GetWakeupFlag(TIMER_T *timer);
static __inline void TIMER_ClearWakeupFlag(TIMER_T *timer);
static __inline uint32_t TIMER_GetCaptureData(TIMER_T *timer);
static __inline uint32_t TIMER_GetCounter(TIMER_T *timer);
static __inline void TIMER_EventCounterSelect(TIMER_T *timer, uint32_t u32Src);
# 259 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_Start(TIMER_T *timer)
{
    timer->CTL |= (0x1ul << (30));
}
# 273 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_Stop(TIMER_T *timer)
{
    timer->CTL &= ~(0x1ul << (30));
}
# 289 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_EnableWakeup(TIMER_T *timer)
{
    timer->CTL |= (0x1ul << (23));
}
# 303 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_DisableWakeup(TIMER_T *timer)
{
    timer->CTL &= ~(0x1ul << (23));
}
# 317 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_StartCapture(TIMER_T *timer)
{
    timer->EXTCTL |= (0x1ul << (3));
}
# 331 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_StopCapture(TIMER_T *timer)
{
    timer->EXTCTL &= ~(0x1ul << (3));
}
# 345 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_EnableCaptureDebounce(TIMER_T *timer)
{
    timer->EXTCTL |= (0x1ul << (6));
}
# 359 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_DisableCaptureDebounce(TIMER_T *timer)
{
    timer->EXTCTL &= ~(0x1ul << (6));
}
# 373 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_EnableEventCounterDebounce(TIMER_T *timer)
{
    timer->EXTCTL |= (0x1ul << (7));
}
# 387 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_DisableEventCounterDebounce(TIMER_T *timer)
{
    timer->EXTCTL &= ~(0x1ul << (7));
}
# 401 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_EnableInt(TIMER_T *timer)
{
    timer->CTL |= (0x1ul << (29));
}
# 415 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_DisableInt(TIMER_T *timer)
{
    timer->CTL &= ~(0x1ul << (29));
}
# 429 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_EnableCaptureInt(TIMER_T *timer)
{
    timer->EXTCTL |= (0x1ul << (5));
}
# 443 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_DisableCaptureInt(TIMER_T *timer)
{
    timer->EXTCTL &= ~(0x1ul << (5));
}
# 458 "../../../../Library/StdDriver/inc\\timer.h"
static __inline uint32_t TIMER_GetIntFlag(TIMER_T *timer)
{
    return ((timer->INTSTS & (0x1ul << (0))) ? 1UL : 0UL);
}
# 472 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_ClearIntFlag(TIMER_T *timer)
{
    timer->INTSTS = (0x1ul << (0));
}
# 487 "../../../../Library/StdDriver/inc\\timer.h"
static __inline uint32_t TIMER_GetCaptureIntFlag(TIMER_T *timer)
{
    return timer->EINTSTS;
}
# 502 "../../../../Library/StdDriver/inc\\timer.h"
static __inline uint32_t TIMER_GetCaptureIntFlagOV(TIMER_T *timer)
{
    return ((timer->EINTSTS & (0x1ul << (1))) ? 1UL : 0UL);
}
# 516 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_ClearCaptureIntFlag(TIMER_T *timer)
{
    timer->EINTSTS = (0x1ul << (0));
}
# 531 "../../../../Library/StdDriver/inc\\timer.h"
static __inline uint32_t TIMER_GetWakeupFlag(TIMER_T *timer)
{
    return (timer->INTSTS & (0x1ul << (1)) ? 1UL : 0UL);
}
# 545 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_ClearWakeupFlag(TIMER_T *timer)
{
    timer->INTSTS = (0x1ul << (1));
}
# 559 "../../../../Library/StdDriver/inc\\timer.h"
static __inline uint32_t TIMER_GetCaptureData(TIMER_T *timer)
{
    return timer->CAP;
}
# 573 "../../../../Library/StdDriver/inc\\timer.h"
static __inline uint32_t TIMER_GetCounter(TIMER_T *timer)
{
    return timer->CNT;
}
# 593 "../../../../Library/StdDriver/inc\\timer.h"
static __inline void TIMER_EventCounterSelect(TIMER_T *timer, uint32_t u32Src)
{
    timer->EXTCTL = (timer->EXTCTL & ~(0x7ul << (16))) | u32Src;
}

uint32_t TIMER_Open(TIMER_T *timer, uint32_t u32Mode, uint32_t u32Freq);
void TIMER_Close(TIMER_T *timer);
int32_t TIMER_Delay(TIMER_T *timer, uint32_t u32Usec);
void TIMER_EnableCapture(TIMER_T *timer, uint32_t u32CapMode, uint32_t u32Edge);
void TIMER_CaptureSelect(TIMER_T *timer, uint32_t u32Src);
void TIMER_DisableCapture(TIMER_T *timer);
void TIMER_EnableEventCounter(TIMER_T *timer, uint32_t u32Edge);
void TIMER_DisableEventCounter(TIMER_T *timer);
uint32_t TIMER_GetModuleClock(TIMER_T *timer);
void TIMER_EnableFreqCounter(TIMER_T *timer,
                             uint32_t u32DropCount,
                             uint32_t u32Timeout,
                             uint32_t u32EnableInt);
void TIMER_DisableFreqCounter(TIMER_T *timer);
void TIMER_SetTriggerSource(TIMER_T *timer, uint32_t u32Src);
void TIMER_SetTriggerTarget(TIMER_T *timer, uint32_t u32Mask);
int32_t TIMER_ResetCounter(TIMER_T *timer);
void TIMER_EnableCaptureInputNoiseFilter(TIMER_T *timer, uint32_t u32FilterCount, uint32_t u32ClkSrcSel);
void TIMER_DisableCaptureInputNoiseFilter(TIMER_T *timer);
# 1741 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\timer_pwm.h" 1
# 761 "../../../../Library/StdDriver/inc\\timer_pwm.h"
void TPWM_SetCounterClockSource(TIMER_T *timer, uint32_t u32CntClkSrc);
uint32_t TPWM_ConfigOutputFreqAndDuty(TIMER_T *timer, uint32_t u32Frequency, uint32_t u32DutyCycle);
void TPWM_EnableDeadTime(TIMER_T *timer, uint32_t u32DTCount);
void TPWM_EnableDeadTimeWithPrescale(TIMER_T *timer, uint32_t u32DTCount);
void TPWM_DisableDeadTime(TIMER_T *timer);
void TPWM_EnableCounter(TIMER_T *timer);
void TPWM_DisableCounter(TIMER_T *timer);
void TPWM_EnableTriggerEADC(TIMER_T *timer, uint32_t u32Condition);
void TPWM_DisableTriggerEADC(TIMER_T *timer);
void TPWM_EnableFaultBrake(TIMER_T *timer, uint32_t u32CH0Level, uint32_t u32CH1Level, uint32_t u32BrakeSource);
void TPWM_EnableFaultBrakeInt(TIMER_T *timer, uint32_t u32IntSource);
void TPWM_DisableFaultBrakeInt(TIMER_T *timer, uint32_t u32IntSource);
uint32_t TPWM_GetFaultBrakeIntFlag(TIMER_T *timer, uint32_t u32IntSource);
void TPWM_ClearFaultBrakeIntFlag(TIMER_T *timer, uint32_t u32IntSource);
void TPWM_SetLoadMode(TIMER_T *timer, uint32_t u32LoadMode);
void TPWM_EnableBrakePinDebounce(TIMER_T *timer, uint32_t u32BrakePinSrc, uint32_t u32DebounceCnt, uint32_t u32ClkSrcSel);
void TPWM_DisableBrakePinDebounce(TIMER_T *timer);
void TPWM_EnableBrakePinInverse(TIMER_T *timer);
void TPWM_DisableBrakePinInverse(TIMER_T *timer);
void TPWM_SetBrakePinSource(TIMER_T *timer, uint32_t u32BrakePinNum);
void TPWM_EnableAcc(TIMER_T *timer, uint32_t u32IntFlagCnt, uint32_t u32IntAccSrc);
void TPWM_DisableAcc(TIMER_T *timer);
void TPWM_EnableAccInt(TIMER_T *timer);
void TPWM_DisableAccInt(TIMER_T *timer);
void TPWM_ClearAccInt(TIMER_T *timer);
uint32_t TPWM_GetAccInt(TIMER_T *timer);
void TPWM_EnableAccPDMA(TIMER_T *timer);
void TPWM_DisableAccPDMA(TIMER_T *timer);
void TPWM_EnableAccStopMode(TIMER_T *timer);
void TPWM_DisableAccStopMode(TIMER_T *timer);
void TPWM_EnableExtEventTrigger(TIMER_T *timer, uint32_t u32ExtEventSrc, uint32_t u32CounterAction);
void TPWM_DisableExtEventTrigger(TIMER_T *timer);
# 1742 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\trng.h" 1
# 81 "../../../../Library/StdDriver/inc\\trng.h"
int32_t TRNG_Open(void);
int32_t TRNG_GenWord(uint32_t *u32RndNum);
int32_t TRNG_GenBignum(uint8_t u8BigNum[], int32_t i32Len);
int32_t TRNG_GenBignumHex(char cBigNumHex[], int32_t i32Len);
# 1743 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\ttmr.h" 1
# 112 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_Start(TTMR_T *ttmr);
static __inline void TTMR_Stop(TTMR_T *ttmr);
static __inline void TTMR_EnableWakeup(TTMR_T *ttmr);
static __inline void TTMR_DisableWakeup(TTMR_T *ttmr);
static __inline void TTMR_EnableInt(TTMR_T *ttmr);
static __inline void TTMR_DisableInt(TTMR_T *ttmr);
static __inline uint32_t TTMR_GetIntFlag(TTMR_T *ttmr);
static __inline void TTMR_ClearIntFlag(TTMR_T *ttmr);
static __inline uint32_t TTMR_GetWakeupFlag(TTMR_T *ttmr);
static __inline void TTMR_ClearWakeupFlag(TTMR_T *ttmr);
static __inline uint32_t TTMR_GetCounter(TTMR_T *ttmr);
static __inline void TTMR_EnablePDCLK(TTMR_T *ttmr);
static __inline void TTMR_DisablePDCLK(TTMR_T *ttmr);
# 135 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_Start(TTMR_T *ttmr)
{
    ttmr->CTL |= (0x1ul << (30));
}
# 149 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_Stop(TTMR_T *ttmr)
{
    ttmr->CTL &= ~(0x1ul << (30));
}
# 165 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_EnableWakeup(TTMR_T *ttmr)
{
    ttmr->CTL |= ((0x1ul << (23)) | (0x1UL << (16)));
}
# 179 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_DisableWakeup(TTMR_T *ttmr)
{
    ttmr->CTL &= ~(0x1ul << (23));
}
# 194 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_EnableInt(TTMR_T *ttmr)
{
    ttmr->CTL |= (0x1ul << (29));
}
# 208 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_DisableInt(TTMR_T *ttmr)
{
    ttmr->CTL &= ~(0x1ul << (29));
}
# 224 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline uint32_t TTMR_GetIntFlag(TTMR_T *ttmr)
{
    return ((ttmr->INTSTS & (0x1ul << (0))) ? 1UL : 0UL);
}
# 238 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_ClearIntFlag(TTMR_T *ttmr)
{
    ttmr->INTSTS = (0x1ul << (0));
}
# 254 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline uint32_t TTMR_GetWakeupFlag(TTMR_T *ttmr)
{
    return (ttmr->INTSTS & (0x1ul << (1)) ? 1UL : 0UL);
}
# 268 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_ClearWakeupFlag(TTMR_T *ttmr)
{
    ttmr->INTSTS = (0x1ul << (1));
}
# 283 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline uint32_t TTMR_GetCounter(TTMR_T *ttmr)
{
    return ttmr->CNT;
}
# 297 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_EnablePDCLK(TTMR_T *ttmr)
{
    ttmr->CTL |= (0x1UL << (16));
}
# 311 "../../../../Library/StdDriver/inc\\ttmr.h"
static __inline void TTMR_DisablePDCLK(TTMR_T *ttmr)
{
    ttmr->CTL &= ~(0x1UL << (16));
}

uint32_t TTMR_Open(TTMR_T *ttmr, uint32_t u32Mode, uint32_t u32Freq);
void TTMR_Close(TTMR_T *ttmr);
int32_t TTMR_Delay(TTMR_T *ttmr, uint32_t u32Usec);
uint32_t TTMR_GetModuleClock(TTMR_T *ttmr);
void TTMR_SetTriggerTarget(TTMR_T *ttmr, uint32_t u32Mask);
int32_t TTMR_ResetCounter(TTMR_T *ttmr);
# 1744 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\uart.h" 1
# 404 "../../../../Library/StdDriver/inc\\uart.h"
static __inline void UART_CLEAR_RTS(UART_T *uart);
static __inline void UART_SET_RTS(UART_T *uart);
# 417 "../../../../Library/StdDriver/inc\\uart.h"
static __inline void UART_CLEAR_RTS(UART_T *uart)
{
    uart->MODEM |= (0x1ul << (9));
    uart->MODEM &= ~(0x1ul << (1));
}
# 433 "../../../../Library/StdDriver/inc\\uart.h"
static __inline void UART_SET_RTS(UART_T *uart)
{
    uart->MODEM |= (0x1ul << (9)) | (0x1ul << (1));
}
# 566 "../../../../Library/StdDriver/inc\\uart.h"
void UART_ClearIntFlag(UART_T *uart, uint32_t u32InterruptFlag);
void UART_Close(UART_T *uart);
void UART_DisableFlowCtrl(UART_T *uart);
void UART_DisableInt(UART_T *uart, uint32_t u32InterruptFlag);
void UART_EnableFlowCtrl(UART_T *uart);
void UART_EnableInt(UART_T *uart, uint32_t u32InterruptFlag);
void UART_Open(UART_T *uart, uint32_t u32baudrate);
uint32_t UART_Read(UART_T *uart, uint8_t pu8RxBuf[], uint32_t u32ReadBytes);
void UART_SetLineConfig(UART_T *uart, uint32_t u32baudrate, uint32_t u32data_width, uint32_t u32parity, uint32_t u32stop_bits);
void UART_SetTimeoutCnt(UART_T *uart, uint32_t u32TOC);
void UART_SelectIrDAMode(UART_T *uart, uint32_t u32Buadrate, uint32_t u32Direction);
void UART_SelectRS485Mode(UART_T *uart, uint32_t u32Mode, uint32_t u32Addr);
void UART_SelectLINMode(UART_T *uart, uint32_t u32Mode, uint32_t u32BreakLength);
uint32_t UART_Write(UART_T *uart, uint8_t pu8TxBuf[], uint32_t u32WriteBytes);
void UART_SelectSingleWireMode(UART_T *uart);
void UART_SetBaudRateFrationalDivider(UART_T *uart, uint32_t u32BRFD);
void UART_DisableBaudRateFrationalDivider(UART_T *uart);
# 1745 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\utcpd.h" 1
# 12 "../../../../Library/StdDriver/inc\\utcpd.h"
# 1 "../../../../Library/Device/Nuvoton/M55M1/Include\\NuMicro.h" 1
# 13 "../../../../Library/StdDriver/inc\\utcpd.h" 2
# 288 "../../../../Library/StdDriver/inc\\utcpd.h"
int32_t UTCPD_GetAlertStatus(int port, int *i32AlertSts);
int32_t UTCPD_ClearAlertStatus(int port, int AlertStClr);
int32_t UTCPD_EnableAlertMask(int port, int mask_set);
int32_t UTCPD_DisableAlertMask(int port, int mask_clr);
int32_t UTCPD_EnablePowerStatusMask(int port, int mask_set);
int32_t UTCPD_DisablePowerStatusMask(int port, int mask_clr);
int32_t UTCPD_EnableFaultMask(int port, int mask_set);
int32_t UTCPD_DisableFaultMask(int port, int mask_clr);
int32_t UTCPD_EnablePowerCtrl(int port, uint32_t mask_set);
int32_t UTCPD_DisablePowerCtrl(int port, uint32_t maskclr);
int32_t UTCPD_EnableFaultCtrl(int port, uint32_t mask_set);
int32_t UTCPD_DisableFaultCtrl(int port, uint32_t maskclr);
int32_t UTCPD_ClearPowerStatus(int port, int PowerStClr);
int32_t UTCPD_EnableFaultStatusMask(int port, int mask_set);
int32_t UTCPD_DisableFaultStatusMask(int port, int mask_clr);
int32_t UTCPD_GetFaultStatus(int port, int *pi32RegData);
int32_t UTCPD_ClearFaultStatus(int port, int FaultStClr);
int32_t UTCPD_SetRoleCtrl(int port, uint32_t u32DrpToggle, uint32_t u32Rpvalue, uint32_t u32CC2, uint32_t u32CC1);
int32_t UTCPD_GetRoleCtrl(int port, uint32_t *pu32DrpToggle, uint32_t *pu32CC1, uint32_t *pu32CC2, uint32_t *pu32Rpvalue);
int32_t UTCPD_SetTypeCPortCtrl(int port, uint32_t u32BistMode, uint32_t u32Orient);
int32_t UTCPD_GetTypeCPortCtrl(int port, uint32_t *pu32BistMode, uint32_t *pu32Orient);
int32_t UTCPD_IsssueCmd(int port, uint32_t cmd);
int32_t UTCPD_GetCCSts(int port, uint32_t *pu32Look4Con, uint32_t *pu32ConRlt, uint32_t *pu32CC2Sts, uint32_t *pu32CC1Sts);
int32_t UTCPD_GetPwrSts(int port, uint32_t *pu32VBUSDetEn, uint32_t *pu32VBUSPresent, uint32_t *pu32VCONNPresent, uint32_t *pu32SnkVBUS);
int32_t UTCPD_GetPwrStsExt(int port, uint32_t *pu32DbgAccessory, uint32_t *pu32SrcNonDefVBUS, uint32_t *pu32SrcDefVBUS);
int32_t UTCPD_GetFaultSts(int port, uint32_t *pu32VBUSOverCurr, uint32_t *pu32VBUSOverVolt, uint32_t *pu32VCONNOverCurr, uint32_t *pu32I2CInfErr);
int32_t UTCPD_GetFaultStsExt(int port, uint32_t *pu32ForceOffFat, uint32_t *pu32AutoDiscFat, uint32_t *pu32ForceDiscFat);
int32_t UTCPD_SetMsgHeaderInfo(int port, uint32_t u32DataRole, uint32_t u32Revision, uint32_t u32PwrRole);
int32_t UTCPD_SetRecDetect(int port, uint32_t u32RegData);
int32_t UTCPD_SetVBUSAlarm(int port, uint32_t u32AlarmH, uint32_t u32AlarmL);
int32_t UTCPD_SetSnkDisconnect(int port, uint32_t u32SnkDiscVolt);
int32_t UTCPD_SetStopDischargeVolt(int port, uint32_t u32StopDischgVolt);
void UTCPD_vconn_disable_src_cc(int port);
void UTCPD_vconn_enable_src_cc(int port);
void UTCPD_vconn_enable_from_cc2(int port);
void UTCPD_vconn_enable_from_cc1(int port);
void UTCPD_vconn_polarity_active_low(int port);
void UTCPD_vconn_polarity_active_high(int port);
void UTCPD_vconn_disable_oc_fault(int port);
void UTCPD_vconn_enable_oc_fault(int port);
void UTCPD_vconn_mux_selection(int port, uint32_t cc1vcensel, uint32_t cc2vcensel);
void UTCPD_vconn_configure_oc_detection_soruce(int port, uint32_t u32Src);
void UTCPD_vbus_srcen_polarity_active_low(int port);
void UTCPD_vbus_srcen_polarity_active_high(int port);
void UTCPD_vbus_snken_polarity_active_low(int port);
void UTCPD_vbus_snken_polarity_active_high(int port);
void UTCPD_vbus_disable_oc_fault(int port);
void UTCPD_vbus_enable_oc_fault(int port);
void UTCPD_vbus_discharge_polarity_active_low(int port);
void UTCPD_vbus_discharge_polarity_active_high(int port);
void UTCPD_vbus_configure_oc_soruce(int port, uint32_t u32Src);
void UTCPD_vbus_disable_ov_fault(int port);
void UTCPD_vbus_enable_ov_fault(int port);
void UTCPD_vbus_disable_forceoff_fault(int port);
void UTCPD_vbus_enable_forceoff_fault(int port);
uint32_t UTCPD_vbus_is_source(int port);
uint32_t UTCPD_vbus_is_sink(int port);
uint32_t UTCPD_vbus_is_source_hv(int port);
void UTCPD_power_enable_monitor(int port);
void UTCPD_power_disable_monitor(int port);
void UTCPD_power_disable_auto_discharge(int port);
void UTCPD_power_enable_auto_discharge(int port);
void UTCPD_frs_tx_polarity_active_low(int port);
void UTCPD_frs_tx_polarity_active_high(int port);
void UTCPD_frs_mux_selection(int port, uint32_t cc1frssel, uint32_t cc2frssel);
uint32_t UTCPD_Open(int port);
# 1746 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\usbd.h" 1
# 32 "../../../../Library/StdDriver/inc\\usbd.h"
typedef struct s_usbd_info
{
    uint8_t *gu8DevDesc;
    uint8_t *gu8ConfigDesc;
    uint8_t **gu8StringDesc;
    uint8_t **gu8HidReportDesc;
    uint8_t *gu8BosDesc;
    uint32_t *gu32HidReportSize;
    uint32_t *gu32ConfigHidDescIdx;

} S_USBD_INFO_T;

extern const S_USBD_INFO_T gsInfo;
# 686 "../../../../Library/StdDriver/inc\\usbd.h"
static __inline void USBD_MemCopy(uint8_t dest[], uint8_t src[], uint32_t size)
{
    uint32_t i = 0ul;

    while (i < size)
    {
        dest[i] = src[i];
        i++;
    }
}
# 707 "../../../../Library/StdDriver/inc\\usbd.h"
static __inline void USBD_SetStall(uint8_t epnum)
{
    uint32_t u32CfgAddr;
    uint32_t u32Cfg;
    uint32_t i;

    for (i = 0ul; i < 25ul; i++)
    {
        u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&((USBD_T *) ((((uint32_t) 0x40000000UL) + 0x00250000UL) + 0x0C000UL))->EP[0].CFG;
        u32Cfg = *((volatile uint32_t *)(u32CfgAddr));

        if ((u32Cfg & 0xful) == epnum)
        {
            u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&((USBD_T *) ((((uint32_t) 0x40000000UL) + 0x00250000UL) + 0x0C000UL))->EP[0].CFGP;
            u32Cfg = *((volatile uint32_t *)(u32CfgAddr));

            *((volatile uint32_t *)(u32CfgAddr)) = (u32Cfg | (0x1ul << (1)));
            break;
        }
    }
}
# 739 "../../../../Library/StdDriver/inc\\usbd.h"
static __inline void USBD_ClearStall(uint8_t epnum)
{
    uint32_t u32CfgAddr;
    uint32_t u32Cfg;
    uint32_t i;

    for (i = 0ul; i < 25ul; i++)
    {
        u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&((USBD_T *) ((((uint32_t) 0x40000000UL) + 0x00250000UL) + 0x0C000UL))->EP[0].CFG;
        u32Cfg = *((volatile uint32_t *)(u32CfgAddr));

        if ((u32Cfg & 0xful) == epnum)
        {
            u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&((USBD_T *) ((((uint32_t) 0x40000000UL) + 0x00250000UL) + 0x0C000UL))->EP[0].CFGP;
            u32Cfg = *((volatile uint32_t *)(u32CfgAddr));

            *((volatile uint32_t *)(u32CfgAddr)) = (u32Cfg & ~(0x1ul << (1)));
            break;
        }
    }
}
# 772 "../../../../Library/StdDriver/inc\\usbd.h"
static __inline uint32_t USBD_GetStall(uint8_t epnum)
{
    uint32_t u32CfgAddr;
    uint32_t u32Cfg;
    uint32_t i;

    for (i = 0ul; i < 25ul; i++)
    {
        u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&((USBD_T *) ((((uint32_t) 0x40000000UL) + 0x00250000UL) + 0x0C000UL))->EP[0].CFG;
        u32Cfg = *((volatile uint32_t *)(u32CfgAddr));

        if ((u32Cfg & 0xful) == epnum)
        {
            u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&((USBD_T *) ((((uint32_t) 0x40000000UL) + 0x00250000UL) + 0x0C000UL))->EP[0].CFGP;
            break;
        }
    }

    return ((*((volatile uint32_t *)(u32CfgAddr))) & (0x1ul << (1)));
}

extern uint8_t g_usbd_SetupPacket[8];
extern volatile uint8_t g_usbd_RemoteWakeupEn;

typedef void (*VENDOR_REQ)(void);
typedef void (*CLASS_REQ)(void);
typedef void (*SET_INTERFACE_REQ)(uint32_t u32AltInterface);
typedef void (*SET_CONFIG_CB)(void);


void USBD_Open(const S_USBD_INFO_T *param, CLASS_REQ pfnClassReq, SET_INTERFACE_REQ pfnSetInterface);
void USBD_Start(void);
void USBD_GetSetupPacket(uint8_t *buf);
void USBD_ProcessSetupPacket(void);
void USBD_GetDescriptor(void);
void USBD_StandardRequest(void);
void USBD_PrepareCtrlIn(uint8_t pu8Buf[], uint32_t u32Size);
void USBD_CtrlIn(void);
void USBD_PrepareCtrlOut(uint8_t *pu8Buf, uint32_t u32Size);
void USBD_CtrlOut(void);
void USBD_SwReset(void);
void USBD_SetVendorRequest(VENDOR_REQ pfnVendorReq);
void USBD_SetConfigCallback(SET_CONFIG_CB pfnSetConfigCallback);
void USBD_LockEpStall(uint32_t u32EpBitmap);
# 1747 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\usci_i2c.h" 1
# 33 "../../../../Library/StdDriver/inc\\usci_i2c.h"
enum UI2C_MASTER_EVENT
{
    MASTER_SEND_ADDRESS = 10,
    MASTER_SEND_H_WR_ADDRESS,
    MASTER_SEND_H_RD_ADDRESS,
    MASTER_SEND_L_ADDRESS,
    MASTER_SEND_DATA,
    MASTER_SEND_REPEAT_START,
    MASTER_READ_DATA,
    MASTER_STOP,
    MASTER_SEND_START
};




enum UI2C_SLAVE_EVENT
{
    SLAVE_ADDRESS_ACK = 100,
    SLAVE_H_WR_ADDRESS_ACK,
    SLAVE_L_WR_ADDRESS_ACK,
    SLAVE_GET_DATA,
    SLAVE_SEND_DATA,
    SLAVE_H_RD_ADDRESS_ACK,
    SLAVE_L_RD_ADDRESS_ACK
};
# 101 "../../../../Library/StdDriver/inc\\usci_i2c.h"
extern int32_t g_UI2C_i32ErrCode;
# 283 "../../../../Library/StdDriver/inc\\usci_i2c.h"
uint32_t UI2C_Open(UI2C_T *ui2c, uint32_t u32BusClock);
void UI2C_Close(UI2C_T *ui2c);
void UI2C_ClearTimeoutFlag(UI2C_T *ui2c);
void UI2C_Trigger(UI2C_T *ui2c, uint8_t u8Start, uint8_t u8Stop, uint8_t u8Ptrg, uint8_t u8Ack);
void UI2C_DisableInt(UI2C_T *ui2c, uint32_t u32Mask);
void UI2C_EnableInt(UI2C_T *ui2c, uint32_t u32Mask);
uint32_t UI2C_GetBusClockFreq(UI2C_T *ui2c);
uint32_t UI2C_SetBusClockFreq(UI2C_T *ui2c, uint32_t u32BusClock);
uint32_t UI2C_GetIntFlag(UI2C_T *ui2c, uint32_t u32Mask);
void UI2C_ClearIntFlag(UI2C_T *ui2c, uint32_t u32Mask);
uint32_t UI2C_GetData(UI2C_T *ui2c);
void UI2C_SetData(UI2C_T *ui2c, uint8_t u8Data);
void UI2C_SetSlaveAddr(UI2C_T *ui2c, uint8_t u8SlaveNo, uint16_t u16SlaveAddr, uint8_t u8GCMode);
void UI2C_SetSlaveAddrMask(UI2C_T *ui2c, uint8_t u8SlaveNo, uint16_t u16SlaveAddrMask);
void UI2C_EnableTimeout(UI2C_T *ui2c, uint32_t u32TimeoutCnt);
void UI2C_DisableTimeout(UI2C_T *ui2c);
void UI2C_EnableWakeup(UI2C_T *ui2c, uint8_t u8WakeupMode);
void UI2C_DisableWakeup(UI2C_T *ui2c);
uint8_t UI2C_WriteByte(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint8_t data);
uint32_t UI2C_WriteMultiBytes(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint8_t *data, uint32_t u32wLen);
uint8_t UI2C_WriteByteOneReg(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr, uint8_t data);
uint32_t UI2C_WriteMultiBytesOneReg(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr, uint8_t *data, uint32_t u32wLen);
uint8_t UI2C_WriteByteTwoRegs(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr, uint8_t data);
uint32_t UI2C_WriteMultiBytesTwoRegs(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr, uint8_t *data, uint32_t u32wLen);
uint8_t UI2C_ReadByte(UI2C_T *ui2c, uint8_t u8SlaveAddr);
uint32_t UI2C_ReadMultiBytes(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint8_t *rdata, uint32_t u32rLen);
uint8_t UI2C_ReadByteOneReg(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr);
uint32_t UI2C_ReadMultiBytesOneReg(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint8_t u8DataAddr, uint8_t *rdata, uint32_t u32rLen);
uint8_t UI2C_ReadByteTwoRegs(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr);
uint32_t UI2C_ReadMultiBytesTwoRegs(UI2C_T *ui2c, uint8_t u8SlaveAddr, uint16_t u16DataAddr, uint8_t *rdata, uint32_t u32rLen);
# 1748 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\usci_spi.h" 1
# 396 "../../../../Library/StdDriver/inc\\usci_spi.h"
uint32_t USPI_Open(USPI_T *uspi, uint32_t u32MasterSlave, uint32_t u32SPIMode, uint32_t u32DataWidth, uint32_t u32BusClock);
void USPI_Close(USPI_T *uspi);
void USPI_ClearRxBuf(USPI_T *uspi);
void USPI_ClearTxBuf(USPI_T *uspi);
void USPI_DisableAutoSS(USPI_T *uspi);
void USPI_EnableAutoSS(USPI_T *uspi, uint32_t u32SSPinMask, uint32_t u32ActiveLevel);
uint32_t USPI_SetBusClock(USPI_T *uspi, uint32_t u32BusClock);
uint32_t USPI_GetBusClock(USPI_T *uspi);
void USPI_EnableInt(USPI_T *uspi, uint32_t u32Mask);
void USPI_DisableInt(USPI_T *uspi, uint32_t u32Mask);
uint32_t USPI_GetIntFlag(USPI_T *uspi, uint32_t u32Mask);
void USPI_ClearIntFlag(USPI_T *uspi, uint32_t u32Mask);
uint32_t USPI_GetStatus(USPI_T *uspi, uint32_t u32Mask);
void USPI_EnableWakeup(USPI_T *uspi);
void USPI_DisableWakeup(USPI_T *uspi);
# 1749 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\usci_uart.h" 1
# 513 "../../../../Library/StdDriver/inc\\usci_uart.h"
void UUART_ClearIntFlag(UUART_T *uuart, uint32_t u32Mask);
uint32_t UUART_GetIntFlag(UUART_T *uuart, uint32_t u32Mask);
void UUART_Close(UUART_T *uuart);
void UUART_DisableInt(UUART_T *uuart, uint32_t u32Mask);
void UUART_EnableInt(UUART_T *uuart, uint32_t u32Mask);
uint32_t UUART_Open(UUART_T *uuart, uint32_t u32baudrate);
uint32_t UUART_Read(UUART_T *uuart, uint8_t pu8RxBuf[], uint32_t u32ReadBytes);
uint32_t UUART_SetLine_Config(UUART_T *uuart, uint32_t u32baudrate, uint32_t u32data_width, uint32_t u32parity, uint32_t u32stop_bits);
uint32_t UUART_Write(UUART_T *uuart, uint8_t pu8TxBuf[], uint32_t u32WriteBytes);
void UUART_EnableWakeup(UUART_T *uuart, uint32_t u32WakeupMode);
void UUART_DisableWakeup(UUART_T *uuart);
void UUART_EnableFlowCtrl(UUART_T *uuart);
void UUART_DisableFlowCtrl(UUART_T *uuart);
# 1750 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\wdt.h" 1
# 157 "../../../../Library/StdDriver/inc\\wdt.h"
static __inline void WDT_Close(WDT_T *wdt);
static __inline void WDT_EnableInt(WDT_T *wdt);
static __inline void WDT_DisableInt(WDT_T *wdt);
# 169 "../../../../Library/StdDriver/inc\\wdt.h"
static __inline void WDT_Close(WDT_T *wdt)
{
    uint32_t u32TimeOutCnt = SystemCoreClock;

    wdt->CTL = 0UL;

    while (wdt->CTL & (0x1ul << (30)))
    {
        if (--u32TimeOutCnt == 0) break;
    }

}
# 190 "../../../../Library/StdDriver/inc\\wdt.h"
static __inline void WDT_EnableInt(WDT_T *wdt)
{
    wdt->CTL |= (0x1ul << (6));
}
# 203 "../../../../Library/StdDriver/inc\\wdt.h"
static __inline void WDT_DisableInt(WDT_T *wdt)
{

    wdt->CTL &= ~((0x1ul << (6)));
}

int32_t WDT_Open(WDT_T *wdt, uint32_t u32TimeoutInterval, uint32_t u32ResetDelay, uint32_t u32EnableReset, uint32_t u32EnableWakeup);
# 1751 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 1 "../../../../Library/StdDriver/inc\\wwdt.h" 1
# 136 "../../../../Library/StdDriver/inc\\wwdt.h"
void WWDT_Open(WWDT_T *wwdt, uint32_t u32PreScale, uint32_t u32CmpValue, uint32_t u32EnableInt);
# 1752 "../../../../Library/Device/Nuvoton/M55M1/Include\\M55M1.h" 2
# 15 "../../../../Library/Device/Nuvoton/M55M1/Include\\NuMicro.h" 2
# 11 "../Device/ImageSensor/Sensor/SWI2C.c" 2
# 1 "../Device/ImageSensor/Sensor\\SWI2C.h" 1
# 36 "../Device/ImageSensor/Sensor\\SWI2C.h"
typedef void (*PFN_SWI2C_TIMEDELY)(uint32_t);
typedef struct
{
    uint32_t u32SCKPortIndex;
    uint32_t u32SCKPinMask;
    uint32_t u32SDAPortIndex;
    uint32_t u32SDAPinMask;
} S_I2C_Channel;

typedef enum
{
    eDRVGPIO_GPIOA = 0x00,
    eDRVGPIO_GPIOB = 0x40,
    eDRVGPIO_GPIOC = 0x80,
    eDRVGPIO_GPIOD = 0xC0,
    eDRVGPIO_GPIOE = 0x100,
    eDRVGPIO_GPIOF = 0x140,
    eDRVGPIO_GPIOG = 0x180,
    eDRVGPIO_GPIOH = 0x1C0,
    eDRVGPIO_GPIOI = 0x200,
    eDRVGPIO_GPIOJ = 0x240,
} E_DRVGPIO_PORT;
typedef enum
{
    eDRVGPIO_PIN0 = (1),
    eDRVGPIO_PIN1 = (1 << 1),
    eDRVGPIO_PIN2 = (1 << 2),
    eDRVGPIO_PIN3 = (1 << 3),
    eDRVGPIO_PIN4 = (1 << 4),
    eDRVGPIO_PIN5 = (1 << 5),
    eDRVGPIO_PIN6 = (1 << 6),
    eDRVGPIO_PIN7 = (1 << 7),
    eDRVGPIO_PIN8 = (1 << 8),
    eDRVGPIO_PIN9 = (1 << 9),
    eDRVGPIO_PIN10 = (1 << 10),
    eDRVGPIO_PIN11 = (1 << 11),
    eDRVGPIO_PIN12 = (1 << 12),
    eDRVGPIO_PIN13 = (1 << 13),
    eDRVGPIO_PIN14 = (1 << 14),
    eDRVGPIO_PIN15 = (1 << 15)
} E_DRVGPIO_BIT;




int SWI2C_Open(uint32_t u32SCKPortIndex, uint32_t u32SCKPinMask, uint32_t u32SDAPortIndex,
               uint32_t u32SDAPinMask, PFN_SWI2C_TIMEDELY pfntimedelay);
void SWI2C_Close(void);
void SWI2C_SendStart(void);
void SWI2C_SendStop(void);
int SWI2C_WriteByte(uint8_t u8Data, uint8_t u8AckType, uint8_t u8Length);
uint32_t SWI2C_ReadByte(uint8_t u8AckType, uint8_t u8Length);
uint8_t SWI2C_Slave_ReadByte(uint8_t u8AckType);
uint32_t SWI2C_GetVersion(void);



uint8_t SWI2C_Write_8bitSlaveAddr_8bitReg_8bitData(uint8_t uAddr, uint8_t uRegAddr, uint8_t uData);
uint8_t SWI2C_Read_8bitSlaveAddr_8bitReg_8bitData(uint8_t uAddr, uint8_t uRegAddr);



uint8_t SWI2C_Write_8bitSlaveAddr_16bitReg_8bitData(uint8_t uAddr, uint16_t uRegAddr, uint8_t uData);
uint8_t SWI2C_Read_8bitSlaveAddr_16bitReg_8bitData(uint8_t uAddr, uint16_t uRegAddr);
# 12 "../Device/ImageSensor/Sensor/SWI2C.c" 2





static S_I2C_Channel s_sChannel;
static PFN_SWI2C_TIMEDELY pfntimedelay = 0;

void _SWI2C_SDA_SETIN(uint32_t PortIndex, uint32_t PinMask)
{
    GPIO_SetMode((GPIO_T *)(((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + PortIndex), PinMask, 0x0UL);
}

void _SWI2C_SCK_SETOUT(uint32_t PortIndex, uint32_t PinMask)
{
    GPIO_SetMode((GPIO_T *)(((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + PortIndex), PinMask, 0x1UL);
}

void _SWI2C_SDA_SETOUT(uint32_t PortIndex, uint32_t PinMask)
{
    GPIO_SetMode((GPIO_T *)(((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + PortIndex), PinMask, 0x1UL);
}

static void _SWI2C_Delay(
    uint32_t nCount
)
{
    volatile uint32_t i;

    if (pfntimedelay != 0)
    {
        pfntimedelay(nCount);
    }
    else
    {
        for (; nCount != 0; nCount--)
            for (i = 0; i < 200; i++);
    }
}

int SWI2C_Open(
    uint32_t u32SCKPortIndex,
    uint32_t u32SCKPinMask,
    uint32_t u32SDAPortIndex,
    uint32_t u32SDAPinMask,
    PFN_SWI2C_TIMEDELY pfn_SWI2C_Delay
)
{


    s_sChannel.u32SCKPortIndex = u32SCKPortIndex;
    s_sChannel.u32SCKPinMask = u32SCKPinMask;
    s_sChannel.u32SDAPortIndex = u32SDAPortIndex;
    s_sChannel.u32SDAPinMask = u32SDAPinMask;


    _SWI2C_SCK_SETOUT(s_sChannel.u32SCKPortIndex, s_sChannel.u32SCKPinMask);
    _SWI2C_SDA_SETOUT(s_sChannel.u32SDAPortIndex, s_sChannel.u32SDAPinMask);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))|s_sChannel.u32SCKPinMask));
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))|s_sChannel.u32SDAPinMask));
    _SWI2C_Delay(5);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))&(~s_sChannel.u32SCKPinMask)));
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))&(~s_sChannel.u32SDAPinMask)));
    _SWI2C_Delay(5);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))|s_sChannel.u32SCKPinMask));
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))|s_sChannel.u32SDAPinMask));

    pfntimedelay = pfn_SWI2C_Delay;

    return 0;
}

void SWI2C_Close(void)
{

}

void SWI2C_SendStart(void)
{

    _SWI2C_SDA_SETOUT(s_sChannel.u32SDAPortIndex, s_sChannel.u32SDAPinMask);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))|s_sChannel.u32SDAPinMask));
    _SWI2C_Delay(2);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))|s_sChannel.u32SCKPinMask));
    _SWI2C_Delay(2);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))&(~s_sChannel.u32SDAPinMask)));
    _SWI2C_Delay(2);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))&(~s_sChannel.u32SCKPinMask)));
    _SWI2C_Delay(2);
}

void SWI2C_SendStop(void)
{






    _SWI2C_Delay(10);
    _SWI2C_SDA_SETOUT(s_sChannel.u32SDAPortIndex, s_sChannel.u32SDAPinMask);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))&(~s_sChannel.u32SDAPinMask)));
    _SWI2C_Delay(2);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))|s_sChannel.u32SCKPinMask));
    _SWI2C_Delay(2);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))|s_sChannel.u32SDAPinMask));
    _SWI2C_Delay(2);


}



int SWI2C_WriteByte(
    uint8_t u8Data,
    uint8_t u8AckType,
    uint8_t u8Length
)
{
    uint8_t u8DataCount;
    uint32_t u32HoldPinValue;
    _SWI2C_SDA_SETOUT(s_sChannel.u32SDAPortIndex, s_sChannel.u32SDAPinMask);


    for (u8DataCount = 0; u8DataCount < u8Length; u8DataCount++)
    {
        if (u8Data & 0x80)
            (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))|s_sChannel.u32SDAPinMask));
        else
            (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))&(~s_sChannel.u32SDAPinMask)));

        _SWI2C_Delay(3);
        (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))|s_sChannel.u32SCKPinMask));
        u8Data <<= 1;
        _SWI2C_Delay(2);
        (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))&(~s_sChannel.u32SCKPinMask)));
        _SWI2C_Delay(2);
    }


    if (u8AckType == 0)
        return 0;



    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))&(~s_sChannel.u32SDAPinMask)));
    _SWI2C_SDA_SETIN(s_sChannel.u32SDAPortIndex, s_sChannel.u32SDAPinMask);
    _SWI2C_Delay(3);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))|s_sChannel.u32SCKPinMask));
    _SWI2C_Delay(2);
    u32HoldPinValue = (((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x10)+(s_sChannel.u32SDAPortIndex)))))&s_sChannel.u32SDAPinMask);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))&(~s_sChannel.u32SCKPinMask)));
    _SWI2C_Delay(2);
    return (u32HoldPinValue == 0 ? 0xFFFF0001 : 0);
}



uint32_t SWI2C_ReadByte(
    uint8_t u8AckType,
    uint8_t u8Length
)
{
    uint32_t u32Data = 0;
    uint8_t u8DataCount;

    _SWI2C_SDA_SETIN(s_sChannel.u32SDAPortIndex, s_sChannel.u32SDAPinMask);


    for (u8DataCount = 0; u8DataCount < u8Length; u8DataCount++)
    {
        u32Data = u32Data << 1;
        _SWI2C_Delay(3);
        (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))|s_sChannel.u32SCKPinMask));

        if ((((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x10)+(s_sChannel.u32SDAPortIndex)))))&s_sChannel.u32SDAPinMask) == s_sChannel.u32SDAPinMask)
            u32Data = u32Data | 0x01;

        _SWI2C_Delay(2);
        (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))&(~s_sChannel.u32SCKPinMask)));
        _SWI2C_Delay(2);
    }


    if (u8AckType == 0)
        return u32Data;



    _SWI2C_SDA_SETOUT(s_sChannel.u32SDAPortIndex, s_sChannel.u32SDAPinMask);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))|s_sChannel.u32SDAPinMask));
    _SWI2C_Delay(3);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))|s_sChannel.u32SCKPinMask));
    _SWI2C_Delay(2);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SCKPortIndex)))))&(~s_sChannel.u32SCKPinMask)));
    _SWI2C_Delay(2);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))&(~s_sChannel.u32SDAPinMask)));
    return u32Data;
}



uint8_t SWI2C_Slave_ReadByte(
    uint8_t u8AckType
)
{
    uint8_t u8Data, u8DataCount;
    u8Data = 0;


    for (u8DataCount = 0; u8DataCount < 8;)
    {
        if ((((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x10)+(s_sChannel.u32SCKPortIndex)))))&s_sChannel.u32SCKPinMask) == 0)
        {
            _SWI2C_Delay(3);

            if ((((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x10)+(s_sChannel.u32SCKPortIndex)))))&s_sChannel.u32SCKPinMask) == s_sChannel.u32SCKPinMask)
            {

                u8Data = u8Data << 1;

                if ((((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x10)+(s_sChannel.u32SDAPortIndex)))))&s_sChannel.u32SDAPinMask) == s_sChannel.u32SDAPinMask)
                    u8Data = u8Data | 0x01;

                u8DataCount++;
            }
        }
    }


    if (u8AckType == 0)
        return u8Data;



    _SWI2C_Delay(2);
    _SWI2C_SDA_SETOUT(s_sChannel.u32SDAPortIndex, s_sChannel.u32SDAPinMask);
    (*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))=(((*((volatile unsigned int *)((((((uint32_t) 0x40000000UL) + 0x00200000UL) + 0x29000UL) + 0x8 )+(s_sChannel.u32SDAPortIndex)))))|s_sChannel.u32SDAPinMask));
    _SWI2C_Delay(7);
    _SWI2C_SDA_SETIN(s_sChannel.u32SDAPortIndex, s_sChannel.u32SDAPinMask);
    return u8Data;
}



static void Delay(uint32_t nCount)
{
    volatile uint32_t i;

    for (; nCount != 0; nCount--)
        for (i = 0; i < 100; i++);
}


uint8_t SWI2C_Write_8bitSlaveAddr_8bitReg_8bitData(uint8_t uAddr, uint8_t uRegAddr, uint8_t uData)
{

    volatile uint32_t u32Delay = 0x50;
    SWI2C_SendStart();

    while (u32Delay--);

    if ((SWI2C_WriteByte(uAddr, 1, 8) == (0UL)) ||
            (SWI2C_WriteByte(uRegAddr, 1, 8) == (0UL)) ||
            (SWI2C_WriteByte(uData, 1, 8) == (0UL)))
    {
        SWI2C_SendStop();
        return (0UL);
    }

    SWI2C_SendStop();

    if (uRegAddr == 0x12 && (uData & 0x80) != 0)
    {
        Delay(1000);
    }

    return (1UL);
}


uint8_t SWI2C_Read_8bitSlaveAddr_8bitReg_8bitData(uint8_t uAddr, uint8_t uRegAddr)
{
    uint8_t u8Data;


    SWI2C_SendStart();
    SWI2C_WriteByte(uAddr, 1, 8);
    SWI2C_WriteByte(uRegAddr, 1, 8);
    SWI2C_SendStop();


    SWI2C_SendStart();
    SWI2C_WriteByte(uAddr | 0x01, 1, 8);
    u8Data = SWI2C_ReadByte(1, 8);
    SWI2C_SendStop();

    return u8Data;
}


uint8_t SWI2C_Read_8bitSlaveAddr_16bitReg_8bitData(uint8_t uAddr, uint16_t uRegAddr)
{
    uint8_t u8Data;


    SWI2C_SendStart();
    SWI2C_WriteByte(uAddr, 1, 8);
    SWI2C_WriteByte(uRegAddr >> 8, 1, 8);
    SWI2C_WriteByte(uRegAddr & 0xFF, 1, 8);



    SWI2C_SendStart();
    SWI2C_WriteByte(uAddr | 0x01, 1, 8);
    u8Data = SWI2C_ReadByte(1, 8);
    SWI2C_SendStop();

    return u8Data;
}


uint8_t SWI2C_Write_8bitSlaveAddr_16bitReg_8bitData(uint8_t uAddr, uint16_t uRegAddr, uint8_t uData)
{

    volatile uint8_t u32Delay = 0x50;
    SWI2C_SendStart();

    while (u32Delay--);

    if ((SWI2C_WriteByte(uAddr, 1, 8) == (0UL)) ||
            (SWI2C_WriteByte(uRegAddr >> 8, 1, 8) == (0UL)) ||
            (SWI2C_WriteByte(uRegAddr & 0xff, 1, 8) == (0UL)) ||
            (SWI2C_WriteByte(uData, 1, 8) == (0UL)))
    {
        SWI2C_SendStop();
        return (0UL);
    }

    SWI2C_SendStop();

    if (uRegAddr == 0x12 && (uData & 0x80) != 0)
    {
        Delay(1000);
    }

    return (1UL);
}
