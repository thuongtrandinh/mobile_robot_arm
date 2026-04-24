#include <uxr/client/transport.h>
#include <rmw_microxrcedds_c/config.h>

#include "main.h"
#include "cmsis_os.h"

#ifdef RMW_UXRCE_TRANSPORT_CUSTOM

#define UART_DMA_BUFFER_SIZE 2048

static uint8_t dma_buffer[UART_DMA_BUFFER_SIZE];
static volatile size_t dma_head = 0;

bool cubemx_transport_open(struct uxrCustomTransport * transport)
{
    UART_HandleTypeDef * uart = (UART_HandleTypeDef*) transport->args;

    dma_head = 0;

    /* Start DMA circular reception */
    HAL_UART_Receive_DMA(uart, dma_buffer, UART_DMA_BUFFER_SIZE);

    return true;
}

bool cubemx_transport_close(struct uxrCustomTransport * transport)
{
    UART_HandleTypeDef * uart = (UART_HandleTypeDef*) transport->args;

    HAL_UART_DMAStop(uart);

    return true;
}

size_t cubemx_transport_write(struct uxrCustomTransport* transport,
                              const uint8_t * buf, size_t len, uint8_t * err)
{
    UART_HandleTypeDef * uart = (UART_HandleTypeDef*) transport->args;

    if (uart->gState == HAL_UART_STATE_READY)
    {
        if (HAL_UART_Transmit_DMA(uart, (uint8_t*)buf, len) != HAL_OK)
            return 0;

        /* Wait TX complete */
        while (uart->gState != HAL_UART_STATE_READY)
            osDelay(1);

        return len;
    }

    return 0;
}

size_t cubemx_transport_read(struct uxrCustomTransport* transport,
                             uint8_t* buf, size_t len,
                             int timeout, uint8_t* err)
{
    UART_HandleTypeDef * uart = (UART_HandleTypeDef*) transport->args;

    size_t dma_tail;
    size_t bytes_read = 0;

    while (bytes_read == 0 && timeout > 0)
    {
        /* DMA tail is where DMA is currently writing */
        dma_tail = UART_DMA_BUFFER_SIZE - __HAL_DMA_GET_COUNTER(uart->hdmarx);

        /* New data available? */
        while (dma_head != dma_tail && bytes_read < len)
        {
            buf[bytes_read++] = dma_buffer[dma_head];
            dma_head = (dma_head + 1) % UART_DMA_BUFFER_SIZE;
        }

        if (bytes_read > 0) break;

        timeout--;
        osDelay(1);
    }

    return bytes_read;
}

#endif // RMW_UXRCE_TRANSPORT_CUSTOM
