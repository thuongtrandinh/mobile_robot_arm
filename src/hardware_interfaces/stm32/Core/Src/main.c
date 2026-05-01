/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body - Optimized for BTS7960 & micro-ROS (No IMU)
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "data_struct.h"
#include <geometry_msgs/msg/twist.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "FreeRTOS.h"
#include "task.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;

UART_HandleTypeDef huart6;
DMA_HandleTypeDef hdma_usart6_rx;
DMA_HandleTypeDef hdma_usart6_tx;

/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = {
  .name = "defaultTask",
  .stack_size = 4000 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};
/* Definitions for motorTask */
osThreadId_t motorTaskHandle;
const osThreadAttr_t motorTask_attributes = {
  .name = "motorTask",
  .stack_size = 1536 * 4,
  .priority = (osPriority_t) osPriorityHigh,
};
/* USER CODE BEGIN PV */

/* ---------- BIẾN MICRO-ROS ---------- */
rcl_publisher_t joint_pub;
sensor_msgs__msg__JointState joint_msg;
double pos_left = 0, pos_right = 0; // Đơn vị: Radian

rcl_subscription_t cmd_vel_sub;
geometry_msgs__msg__Twist cmd_vel_msg;
rclc_executor_t executor;

// Tên các khớp phải khớp 100% với file URDF của bạn (swapped order: Right, Left)
const char* joint_names[2] = {"left_wheel_joint", "right_wheel_joint"};

/* ---------- BIẾN ENCODER & KINEMATIC DECOUPLING PID ---------- */
float speed_left = 0, speed_right = 0;     // Vận tốc bánh xe (m/s)
float speed_v = 0, speed_w = 0;            // Vận tốc thực tế: v (m/s), ω (rad/s)
float setpoint_v = 0, setpoint_w = 0;      // Setpoint từ cmd_vel: v*, ω*

// Hệ số PID cho Vận tốc dài (Linear Velocity - v)
float Kp_v = 800.0f, Ki_v = 2800.0f, Kd_v = 0.0f;
float Error_v = 0, pre_Error_v = 0, pre_pre_Error_v = 0, u_v = 0, pre_u_v = 0;

// Hệ số PID cho Vận tốc góc (Angular Velocity - w)
float Kp_w = 400.0f, Ki_w = 1200.0f, Kd_w = 0.0f;
float Error_w = 0, pre_Error_w = 0, pre_pre_Error_w = 0, u_w = 0, pre_u_w = 0;

float duty_left = 0, duty_right = 0;      // Tín hiệu PWM ra motor

int32_t enc_left = 0, pre_enc_left = 0, delta_left = 0;
int32_t enc_right = 0, pre_enc_right = 0, delta_right = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM4_Init(void);
static void MX_USART6_UART_Init(void);
void StartDefaultTask(void *argument);
void MotorTask(void *argument);

/* USER CODE BEGIN PFP */
void Kinematic_PID_Calculate(void);
void PWM_Calculate1(void);
void PWM_Calculate2(void);

// Khai báo Transport cho micro-ROS để tránh warning implicitly
bool cubemx_transport_open(struct uxrCustomTransport * transport);
bool cubemx_transport_close(struct uxrCustomTransport * transport);
size_t cubemx_transport_write(struct uxrCustomTransport* transport, const uint8_t * buf, size_t len, uint8_t * err);
size_t cubemx_transport_read(struct uxrCustomTransport* transport, uint8_t* buf, size_t len, int timeout, uint8_t* err);

void * microros_allocate(size_t size, void * state);
void microros_deallocate(void * pointer, void * state);
void * microros_reallocate(void * pointer, size_t size, void * state);
void * microros_zero_allocate(size_t number_of_elements, size_t size_of_element, void * state);

int _gettimeofday(struct timeval *tv, void *tzvp)
{
    uint32_t tick = xTaskGetTickCount();
    tv->tv_sec = tick / 1000;
    tv->tv_usec = (tick % 1000) * 1000;
    return 0;
}

void cmd_vel_callback(const void *msgin)
{
    const geometry_msgs__msg__Twist *msg = (const geometry_msgs__msg__Twist *)msgin;

    float v = msg->linear.x;   // m/s
    float w = msg->angular.z;  // rad/s

    __disable_irq();
    setpoint_v = v;  // Gán trực tiếp vận tốc dài
    setpoint_w = -w;  // Gán trực tiếp vận tốc góc
    __enable_irq();
}
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_TIM1_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_USART6_UART_Init();
  /* USER CODE BEGIN 2 */
  // 1. Khởi động Encoder
  HAL_TIM_Encoder_Start(&htim1, TIM_CHANNEL_ALL); // Bánh trái
  HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL); // Bánh phải

  // 2. Khởi động PWM cho BTS7960
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1); // Right RPWM
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2); // Right LPWM
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3); // Left RPWM
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_4); // Left LPWM

  // 3. Kích hoạt Driver BTS7960 (EN pins = HIGH)
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_9 | GPIO_PIN_10, GPIO_PIN_SET);
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_10 | GPIO_PIN_12, GPIO_PIN_SET);

  // 4. Reset biến đếm và PID (Kinematic Decoupling)
  __HAL_TIM_SET_COUNTER(&htim1, 0);
  __HAL_TIM_SET_COUNTER(&htim4, 0);

  	// Encoder counters
  	enc_right = 0; pre_enc_right = 0; delta_right = 0;
	enc_left = 0;  pre_enc_left = 0;  delta_left = 0;
	
	// Wheel speeds (feedback)
	speed_left = 0.0f; speed_right = 0.0f;
	
	// Robot center speeds (kinematic variables)
	speed_v = 0.0f; speed_w = 0.0f;
	
	// Setpoints from ROS 2
	setpoint_v = 0.0f; setpoint_w = 0.0f;
	
	// Linear velocity PID
	Error_v = 0; pre_Error_v = 0; pre_pre_Error_v = 0;
	u_v = 0; pre_u_v = 0;
	
	// Angular velocity PID
	Error_w = 0; pre_Error_w = 0; pre_pre_Error_w = 0;
	u_w = 0; pre_u_w = 0;
	
	// PWM outputs
	duty_left = 0.0f; duty_right = 0.0f;
  /* USER CODE END 2 */

  /* Init scheduler */
  osKernelInitialize();

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of defaultTask */
  defaultTaskHandle = osThreadNew(StartDefaultTask, NULL, &defaultTask_attributes);

  /* creation of motorTask */
  motorTaskHandle = osThreadNew(MotorTask, NULL, &motorTask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

  /* Start scheduler */
  osKernelStart();

  /* We should never get here as control is now taken by the scheduler */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 100;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 8;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 0;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 65535;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 4;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 999;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 0;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 65535;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim4, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */

}

/**
  * @brief USART6 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART6_UART_Init(void)
{

  /* USER CODE BEGIN USART6_Init 0 */

  /* USER CODE END USART6_Init 0 */

  /* USER CODE BEGIN USART6_Init 1 */

  /* USER CODE END USART6_Init 1 */
  huart6.Instance = USART6;
  huart6.Init.BaudRate = 921600;
  huart6.Init.WordLength = UART_WORDLENGTH_8B;
  huart6.Init.StopBits = UART_STOPBITS_1;
  huart6.Init.Parity = UART_PARITY_NONE;
  huart6.Init.Mode = UART_MODE_TX_RX;
  huart6.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart6.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart6) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART6_Init 2 */

  /* USER CODE END USART6_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);
  /* DMA2_Stream6_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream6_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream6_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, LD5_Pin|LD6_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_9|GPIO_PIN_10, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_10|GPIO_PIN_12, GPIO_PIN_RESET);

  /*Configure GPIO pin : PA0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : LD5_Pin LD6_Pin */
  GPIO_InitStruct.Pin = LD5_Pin|LD6_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : PA9 PA10 */
  GPIO_InitStruct.Pin = GPIO_PIN_9|GPIO_PIN_10;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PC10 PC12 */
  GPIO_InitStruct.Pin = GPIO_PIN_10|GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
/* --- KINEMATIC DECOUPLING PID & PWM FUNCTIONS --- */
/**
 * @brief Thuật toán điều khiển phân ly động học (Kinematic Decoupling Control)
 *        Với Dynamic Scaling & Anti-Windup
 * 
 * Forward Kinematics:
 *   speed_v = (speed_left + speed_right) / 2
 *   speed_w = (speed_right - speed_left) / wheel_base
 * 
 * Inverse Kinematics:
 *   duty_left  = u_v - u_w
 *   duty_right = u_v + u_w
 */
void Kinematic_PID_Calculate(void) {
	// ===== 1. ĐỌC CẢM BIẾN ENCODER (Incremental Encoder Decoding) =====
	enc_left = __HAL_TIM_GET_COUNTER(&htim1);
	delta_left = enc_left - pre_enc_left;
	if(delta_left > 40000) delta_left -= 65536;
	else if(delta_left < -40000) delta_left += 65536;
	speed_left = ((delta_left / PPR) / Ts) * (2.0f * M_PI * wheel_radius);
	pre_enc_left = enc_left;

	enc_right = __HAL_TIM_GET_COUNTER(&htim4);
	delta_right = enc_right - pre_enc_right;
	if(delta_right > 40000) delta_right -= 65536;
	else if(delta_right < -40000) delta_right += 65536;
	speed_right = ((delta_right / PPR) / Ts) * (2.0f * M_PI * wheel_radius);
	pre_enc_right = enc_right;

	// ===== 2. FORWARD KINEMATICS (Tính toán v & ω từ speed_left, speed_right) =====
	speed_v = (speed_left + speed_right) / 2.0f;           // Vận tốc dài (m/s)
	speed_w = (speed_right - speed_left) / wheel_base;     // Vận tốc góc (rad/s)

	// ===== 3. XỬ LÝ LỆNH DỪNG (Stop Command Handler) =====
	if (setpoint_v == 0.0f && setpoint_w == 0.0f) {
		u_v = 0; pre_u_v = 0; pre_Error_v = 0; pre_pre_Error_v = 0;
		u_w = 0; pre_u_w = 0; pre_Error_w = 0; pre_pre_Error_w = 0;
		duty_left = 0; duty_right = 0;
		return;
	}

	// ===== 4. TÍNH TOÁN PID RỜI RẠC CHO VẬN TỐC DÀI (Linear Velocity PID) =====
	Error_v = setpoint_v - speed_v;
	u_v = pre_u_v + Kp_v * (Error_v - pre_Error_v) 
	            + Ki_v * Ts * Error_v 
	            + (Kd_v / Ts) * (Error_v - 2.0f * pre_Error_v + pre_pre_Error_v);

	// ===== 5. TÍNH TOÁN PID RỜI RẠC CHO VẬN TỐC GÓC (Angular Velocity PID) =====
	Error_w = setpoint_w - speed_w;
	u_w = pre_u_w + Kp_w * (Error_w - pre_Error_w) 
	           + Ki_w * Ts * Error_w 
	           + (Kd_w / Ts) * (Error_w - 2.0f * pre_Error_w + pre_pre_Error_w);

	// ===== 6. INVERSE KINEMATICS + DYNAMIC SCALING (Anti-Saturation) =====
	// Quy ước ROS 2: +angular.z => rẽ trái => bánh phải nhanh hơn bánh trái
	// v_left  = v - (w * wheel_base / 2)
	// v_right = v + (w * wheel_base / 2)
	float angular_term = u_w * (wheel_base * 0.5f);
	float raw_duty_left  = u_v - angular_term;
	float raw_duty_right = u_v + angular_term;

	float max_pwm_limit = 699.0f;
	float max_requested = (fabsf(raw_duty_left) > fabsf(raw_duty_right)) 
	                      ? fabsf(raw_duty_left) : fabsf(raw_duty_right);

	if (max_requested > max_pwm_limit) {
		// Tính hệ số thu nhỏ để giữ tỷ lệ
		float scale_factor = max_pwm_limit / max_requested;
		
		// Thu nhỏ cả 2 bánh theo cùng tỷ lệ (bảo toàn quỹ đạo cong)
		duty_left  = raw_duty_left * scale_factor;
		duty_right = raw_duty_right * scale_factor;
		
		// Back-calculation Anti-Windup: Gắn lại u_v, u_w thực tế
		u_v = u_v * scale_factor;
		u_w = u_w * scale_factor;
	} else {
		duty_left  = raw_duty_left;
		duty_right = raw_duty_right;
	}

	// ===== 7. CẬP NHẬT BIẾN TRẠNG THÁI CHO CHU KỲ SAU =====
	pre_pre_Error_v = pre_Error_v; 
	pre_Error_v = Error_v; 
	pre_u_v = u_v;

	pre_pre_Error_w = pre_Error_w; 
	pre_Error_w = Error_w; 
	pre_u_w = u_w;
}

void PWM_Calculate1() { // Trái
	// Thêm deadzone: Nếu duty nhỏ hơn 1% (10/1000) thì ép về 0 để tránh chạy tự động
	if(fabsf(duty_left) < 10.0f){
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
	}else if(duty_left > 0){
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, (uint32_t)duty_left);
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
	}else{
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, (uint32_t)(-duty_left));
	}
}

void PWM_Calculate2() { // Phải
	// Thêm deadzone: Nếu duty nhỏ hơn 1% (10/1000) thì ép về 0 để tránh chạy tự động
	if(fabsf(duty_right) < 10.0f){
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
	}else if(duty_right > 0){
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, (uint32_t)duty_right);
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
	}else{
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, (uint32_t)(-duty_right));
	}
}
/* USER CODE END 4 */

/* USER CODE BEGIN Header_StartDefaultTask */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* USER CODE BEGIN 5 */
	rmw_uros_set_custom_transport(true, (void *) &huart6, cubemx_transport_open, cubemx_transport_close, cubemx_transport_write, cubemx_transport_read);
	// Setup Allocators...
	rcl_allocator_t allocator = rcl_get_default_allocator();

	while(rmw_uros_ping_agent(1000, 1) != RMW_RET_OK) { osDelay(100); }

	rclc_support_t support;
	rcl_node_t node;

	// --- PHẦN KHỞI TẠO DOMAIN ID 36 ---
	rcl_init_options_t init_options = rcl_get_zero_initialized_init_options();
	rcl_init_options_init(&init_options, allocator);
	rcl_init_options_set_domain_id(&init_options, 36);

	rclc_support_init_with_options(&support, 0, NULL, &init_options, &allocator);
	rclc_node_init_default(&node, "stm32_node", "", &support);

	// 1. Init JointState Publisher
	rclc_publisher_init_default(&joint_pub, &node,
		ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState), "joint_states");

	// 2. Init cmd_vel Subscription
	rclc_subscription_init_default(&cmd_vel_sub, &node,
		ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist), "/diff_cont/cmd_vel_unstamped");

	// 2b. Debug publisher removed for performance optimization
	// (Loại bỏ để tối ưu hóa hiệu năng truyền dữ liệu UART/DMA)

	rclc_executor_init(&executor, &support.context, 1, &allocator);
	rclc_executor_add_subscription(&executor, &cmd_vel_sub, &cmd_vel_msg, &cmd_vel_callback, ON_NEW_DATA);

	// 3. Cấu hình bộ nhớ tĩnh cho JointState (Tránh lỗi Memory)
	joint_msg.header.frame_id.data = "base_link";
	joint_msg.name.data = (rosidl_runtime_c__String*) malloc(2 * sizeof(rosidl_runtime_c__String));
	joint_msg.name.size = 2;
	joint_msg.name.capacity = 2;
	for(int i=0; i<2; i++) {
		joint_msg.name.data[i].data = (char*)joint_names[i];
		joint_msg.name.data[i].size = strlen(joint_names[i]);
		joint_msg.name.data[i].capacity = joint_msg.name.data[i].size + 1;
	}
	joint_msg.position.data = (double*) malloc(2 * sizeof(double));
	joint_msg.position.size = 2;
	joint_msg.position.capacity = 2;
	joint_msg.velocity.data = (double*) malloc(2 * sizeof(double));
	joint_msg.velocity.size = 2;
	joint_msg.velocity.capacity = 2;

	rmw_uros_sync_session(1000);

	uint32_t sync_counter = 0;
	
	/* ========== PUBLISH RATE CONFIGURATION ==========
	 * 100Hz TUYỆT ĐỐI (ổn định nhất): Dùng osDelayUntil
	 * Loại bỏ jitter bằng cơ chế delay tuyệt đối thay vì tương đối
	 * ============================================= */
	
	// Khởi tạo biến thời gian cho osDelayUntil (absolute timing)
	uint32_t tick_start = osKernelGetTickCount(); 
	const uint32_t frequency_ms = 10;  // 10ms = 100Hz (chu kỳ chính xác)
	const float dt_100hz = 0.01f;      // Delta time = 0.01s
	
	while (1)
	{

		// --- Đồng bộ Clock với Master --- //
		sync_counter++;
		if(sync_counter >=5000){  // Sync mỗi ~50 giây ở 100Hz
			rmw_uros_sync_session(10);
			sync_counter = 0;
		}
		// --- TÍNH TOÁN ĐỘNG HỌC (KINEMATICS) ---
		// Vận tốc góc (rad/s) = vận tốc dài (m/s) / bán kính (m)
		double wl = (double)(speed_left / wheel_radius);
		double wr = (double)(speed_right / wheel_radius);

		// Tích phân vị trí (rad) = vị trí cũ + (vận tốc * dt)
		// dt_100hz = 0.01s chính xác cho 100Hz
		pos_left  += wl * dt_100hz;
		pos_right += wr * dt_100hz;

		// --- GÁN DỮ LIỆU (KHỚP CẶP RIGHT-RIGHT, LEFT-LEFT) ---
		int64_t time_ns = rmw_uros_epoch_nanos();
		joint_msg.header.stamp.sec = (int32_t)(time_ns / 1000000000);
		joint_msg.header.stamp.nanosec = (uint32_t)(time_ns % 1000000000);

		joint_msg.position.data[0] = pos_right;
		joint_msg.position.data[1] = pos_left;
		joint_msg.velocity.data[0] = wr;
		joint_msg.velocity.data[1] = wl;

		// --- PUBLISH JOINT STATE ---
		// Debug velocity publisher removed for performance optimization
		rcl_publish(&joint_pub, &joint_msg, NULL);
		rclc_executor_spin_some(&executor, RCL_MS_TO_NS(0));

		HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_14);
		
		/* === PUBLISH RATE CONTROL (ABSOLUTE TIMING) ===
		 * osDelayUntil: Bù trừ thời gian xử lý code → 100% chính xác 100Hz
		 * Không phụ thuộc vào thời gian thực thi của micro-ROS operations
		 * Công thức: tick_start += frequency_ms; osDelayUntil(tick_start);
		 */
		tick_start += frequency_ms;  // Tăng tick_start lên 10ms
		osDelayUntil(tick_start);     // Delay cho đến tick_start
	}
  /* USER CODE END 5 */
}

/* USER CODE BEGIN Header_MotorTask */
/* USER CODE END Header_MotorTask */
void MotorTask(void *argument)
{
  /* USER CODE BEGIN MotorTask */
  /* Infinite loop - 100Hz Motor Control Loop */
  for(;;)
  {
	  // 1. Thuật toán điều khiển phân ly động học
	  //    Tính toán u_v, u_w và duty_left, duty_right với Dynamic Scaling
	  Kinematic_PID_Calculate();

	  // 2. Xuất PWM ra Driver BTS7960
	  PWM_Calculate1();
	  PWM_Calculate2();
	  
	  // Đảm bảo chạy chính xác ở 100Hz
	  osDelay(10);
  }
  /* USER CODE END MotorTask */
}

/**
  * @brief  Period elapsed callback in non blocking mode
  * @note   This function is called  when TIM2 interrupt took place, inside
  * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
  * a global variable "uwTick" used as application time base.
  * @param  htim : TIM handle
  * @retval None
  */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* USER CODE BEGIN Callback 0 */

  /* USER CODE END Callback 0 */
  if (htim->Instance == TIM2)
  {
    HAL_IncTick();
  }
  /* USER CODE BEGIN Callback 1 */

  /* USER CODE END Callback 1 */
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
