/*
 * Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* *(.bss.NoInit.activation_buf_sram) */

.sram_hyperram (NOLOAD): {
  __sram_hyperram_start = .;
  . = ALIGN(16);
  *(.noinit.tflm_arena)
  *(.bss.NoInit.activation_buf_sram)
  __sram_hyperram_end = .;
} > SRAM_HYPERRAM
