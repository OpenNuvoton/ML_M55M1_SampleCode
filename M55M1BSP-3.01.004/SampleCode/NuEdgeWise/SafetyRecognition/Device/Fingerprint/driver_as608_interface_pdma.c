#include "driver_as608_interface_pdma.h"
#include <stdarg.h>

#define UART_RX_DMA_CH 0
#define UART_TX_DMA_CH 1

/*---------------------------------------------------------------------------------------------------------*/
/* Global variables                                                                                        */
/*---------------------------------------------------------------------------------------------------------*/
//UART_PDMA
int32_t UART_TEST_LENGTH = 384;
uint8_t SrcArray[384];
uint8_t DestArray[384];
volatile int32_t IntCnt;
volatile int32_t IsTestOver;


/*---------------------------------------------------------------------------------------------------------*/
/* UART Tx PDMA0 Channel Configuration                                                                      */
/*---------------------------------------------------------------------------------------------------------*/
void PDMA0_UART_TxTest(int32_t UART_transfer_LENGTH)
{
    /* UART Tx PDMA0 channel configuration */
    /* Set transfer width (8 bits) and transfer count */
    PDMA_SetTransferCnt(PDMA0, UART_TX_DMA_CH, PDMA_WIDTH_8, UART_transfer_LENGTH);

    /* Set source/destination address and attributes */
    PDMA_SetTransferAddr(PDMA0, UART_TX_DMA_CH, (uint32_t)SrcArray, PDMA_SAR_INC, (uint32_t)&UART1->DAT, PDMA_DAR_FIX);

    /* Set request source; set basic mode. */
    PDMA_SetTransferMode(PDMA0, UART_TX_DMA_CH, PDMA_UART1_TX, FALSE, 0);

    /* Single request type */
    PDMA_SetBurstType(PDMA0, UART_TX_DMA_CH, PDMA_REQ_SINGLE, 0);

    /* Disable table interrupt */
    PDMA_DisableInt(PDMA0, UART_TX_DMA_CH, PDMA_INT_TEMPTY);
}

/*---------------------------------------------------------------------------------------------------------*/
/* UART Rx PDMA0 Channel Configuration                                                                      */
/*---------------------------------------------------------------------------------------------------------*/
void PDMA0_UART_RxTest(int32_t UART_transfer_LENGTH)
{
    /* UART Rx PDMA0 channel configuration */
    /* Set transfer width (8 bits) and transfer count */
    PDMA_SetTransferCnt(PDMA0, UART_RX_DMA_CH, PDMA_WIDTH_8, UART_transfer_LENGTH);

    /* Set source/destination address and attributes */
    PDMA_SetTransferAddr(PDMA0, UART_RX_DMA_CH, (uint32_t)&UART1->DAT, PDMA_SAR_FIX, (uint32_t)DestArray, PDMA_DAR_INC);

    /* Set request source; set basic mode. */
    PDMA_SetTransferMode(PDMA0, UART_RX_DMA_CH, PDMA_UART1_RX, FALSE, 0);

    /* Single request type */
    PDMA_SetBurstType(PDMA0, UART_RX_DMA_CH, PDMA_REQ_SINGLE, 0);

    /* Disable table interrupt */
    PDMA_DisableInt(PDMA0, UART_RX_DMA_CH, PDMA_INT_TEMPTY);
}

/*---------------------------------------------------------------------------------------------------------*/
/* PDMA Callback function                                                                                  */
/*---------------------------------------------------------------------------------------------------------*/
void PDMA0_Callback_0(void)
{
    printf("\tTransfer Done %d!\r", ++IntCnt);

    /* Use PDMA0 to do UART loopback test 10 times */
    if (IntCnt < 10)
    {
        /* UART Tx and Rx PDMA0 configuration */
        PDMA0_UART_TxTest(UART_TEST_LENGTH);
        PDMA0_UART_RxTest(UART_TEST_LENGTH);

        /* Enable UART Tx and Rx PDMA0 function */
        UART_ENABLE_INT(UART1, (UART_INTEN_RXPDMAEN_Msk | UART_INTEN_TXPDMAEN_Msk));
    }
    else
    {
        /* Test is over */
        IsTestOver = TRUE;
    }
}

/**
 * @brief  interface uart init
 * @return status code
 *         - 0 success
 *         - 1 uart init failed
 * @note   none
 */
uint8_t as608_interface_uart_init(void)
{

    /* Unlock protected registers */
    SYS_UnlockReg();

    /* Reset PDMA0 module */
    SYS_ResetModule(SYS_PDMA0RST);


    /* Reset UART1 */
    SYS_ResetModule(SYS_UART1RST);

    /* Configure UART1 and set UART1 Baudrate */
    UART_Open(UART1, 57600);

    /* Lock protected registers */
    SYS_LockReg();

    /* Enable Interrupt */
    /* Enable PDMA0 channel */
    PDMA_Open(PDMA0, (1 << UART_RX_DMA_CH) | (1 << UART_TX_DMA_CH));

    /* UART Tx and Rx PDMA0 configuration */
    /* Initial, the Tx transfer size maybe changed*/
    PDMA0_UART_TxTest(UART_TEST_LENGTH);
    PDMA0_UART_RxTest(UART_TEST_LENGTH);

    /* Enable PDMA0 Transfer Done Interrupt */
    PDMA_EnableInt(PDMA0, UART_RX_DMA_CH, PDMA_INT_TRANS_DONE);
    PDMA_EnableInt(PDMA0, UART_TX_DMA_CH, PDMA_INT_TRANS_DONE);

    /* Enable PDMA0 Transfer Done Interrupt */
    IntCnt = 0;
    IsTestOver = FALSE;
    NVIC_EnableIRQ(PDMA0_IRQn);

    /* Enable UART Tx and Rx PDMA0 function */
    //UART_ENABLE_INT(UART1, UART_INTEN_TXPDMAEN_Msk);
    //UART_ENABLE_INT(UART1, UART_INTEN_RXPDMAEN_Msk);



    /* Todo: add check bit. Now always true*/
    return 0;
}


/**
 * @brief  interface uart deinit
 * @return status code
 *         - 0 success
 *         - 1 uart deinit failed
 * @note   none
 */
uint8_t as608_interface_uart_deinit(void)
{
    UART_Close(UART1);

    /* Todo: add check bit. Now always true*/
    return 0;
}

/**
 * @brief      interface uart read
 * @param[out] *buf points to a data buffer
 * @param[in]  len is the length of the data buffer
 * @return     status code
 *             - 0 success
 *             - 1 read failed
 * @note       none
 */

uint16_t as608_interface_uart_read(uint8_t *buf, uint16_t len)
{
    uint32_t  u32Count;

    //printf("\n Uart Read: %d\n", len);

    for (u32Count = 0ul; u32Count != len; u32Count++)
    {
        if (u32Count < 50)
        {
            buf[u32Count] = DestArray[u32Count];
        }
    }

    return len;


}

/*---------------------------------------------------------------------------------------------------------*/
/* ISR to handle UART Channel 1 interrupt event                                                            */
/*---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------*/
/* ISR to handle PDMA0 interrupt event                                                                     */
/*---------------------------------------------------------------------------------------------------------*/
NVT_ITCM void PDMA0_IRQHandler(void)
{
    /* Get PDMA0 interrupt status */
    uint32_t status = PDMA_GET_INT_STATUS(PDMA0);

    if (status & PDMA_INTSTS_ABTIF_Msk)  /* Target Abort */
    {
        if (PDMA_GET_ABORT_STS(PDMA0) & PDMA_ABTSTS_ABTIF2_Msk)
            IsTestOver = 2;

        PDMA_CLR_ABORT_FLAG(PDMA0, PDMA_GET_ABORT_STS(PDMA0));
    }
    else if (status & PDMA_INTSTS_TDIF_Msk)    /* Transfer Done */
    {
        /* UART Tx PDMA0 transfer done interrupt flag */
        if (PDMA_GET_TD_STS(PDMA0) & (1 << UART_TX_DMA_CH))
        {

            /* Clear PDMA0 transfer done interrupt flag */
            PDMA_CLR_TD_FLAG(PDMA0, (1 << UART_TX_DMA_CH));

            /* Disable UART Tx PDMA0 function */
            UART_DISABLE_INT(UART1, UART_INTEN_TXPDMAEN_Msk);
        }

        /* UART Rx PDMA0 transfer done interrupt flag */
        if (PDMA_GET_TD_STS(PDMA0) & (1 << UART_RX_DMA_CH))
        {
            /* Clear PDMA0 transfer done interrupt flag */
            PDMA_CLR_TD_FLAG(PDMA0, (1 << UART_RX_DMA_CH));

            /* Disable UART Rx PDMA0 function */
            UART_DISABLE_INT(UART1, UART_INTEN_RXPDMAEN_Msk);

            /* Handle PDMA0 transfer done interrupt event */
            //PDMA0_Callback_0();
        }
    }
    else
    {
        printf("unknown interrupt, status=0x%x !!\n", status);
    }
}

/**
 * @brief  interface uart flush
 * @return status code
 *         - 0 success
 *         - 1 uart flush failed
 * @note   none
 */
/* No need to implement because we use hardware UART and it is with hardware FIFO*/

uint8_t as608_interface_uart_flush(void)
{
    memset(SrcArray, 0, sizeof(SrcArray));
    memset(DestArray, 0, sizeof(DestArray));

    return 0;
}


/**
 * @brief     interface uart write
 * @param[in] *buf points to a data buffer
 * @param[in] len is the length of the data buffer
 * @return    status code
 *            - 0 success
 *            - 1 write failed
 * @note      none
 */

uint8_t as608_interface_uart_write(uint8_t *buf, uint16_t len)
{
    uint32_t  u32Count, u32delayno;
    uint32_t  u32Exit = 0ul;

    //printf("\n Uart Write: %d\n", len);
    for (u32Count = 0ul; u32Count != len; u32Count++)
    {
        SrcArray[u32Count] = buf[u32Count];
    }

    PDMA0_UART_TxTest((int32_t)len);
    PDMA0_UART_RxTest(UART_TEST_LENGTH);

    UART_ENABLE_INT(UART1, (UART_INTEN_RXPDMAEN_Msk | UART_INTEN_TXPDMAEN_Msk));
    //UART_ENABLE_INT(UART1, UART_INTEN_TXPDMAEN_Msk);
    /* From here, the device(as608) should received CMD from TX*/
    /* And maseter received result from RX buf*/


    CLK_SysTickDelay(1000); // 1 ms = 1000 us

    return 0;
}

/**
 * @brief     interface delay ms
 * @param[in] ms
 * @note      none
 */
void as608_interface_delay_ms(uint32_t ms)
{
    //Max: 200MHz => 83886us, 180MHz => 93206us
    uint32_t count;

    for (count = 0; count < ms; count++)
        CLK_SysTickDelay(1000); // 1 ms = 1000 us
}

/**
 * @brief     interface print format data
 * @param[in] fmt is the format data
 * @note      none
 */
void as608_interface_debug_print(const char *const fmt, ...)
{
    //printf("%s", (char const *)fmt);
    //printf("\n");


    char str[256];
    va_list args;

    memset((char *)str, 0, sizeof(char) * 256);
    va_start(args, fmt);
    vsnprintf((char *)str, 255, (char const *)fmt, args);
    va_end(args);

    printf("%s", str);

}
