/*
 * Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

.sram_non_cache (NOLOAD): {
  __sram_non_cache_start = .;
  . = ALIGN(32);
  *(.NonCacheable.ZeroInit)
  . = ALIGN(32);
  *(.bss.vram.data)
  . = ALIGN(32);
  *(.noinit.tflm_arena)
  . = ALIGN(32);
  *(.bss.NoInit.activation_buf_sram)
  __sram_non_cache_end = .;
} > SRAM_NON_CACHE
