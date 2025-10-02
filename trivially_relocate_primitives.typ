#import "wg21.typ" as wg21

#show: wg21.template.with(
  paper_number: "DXXXXR0",
  audience: "LEWG",
  title: [Tweak trivial relocation library primitives \[DRAFT\] ],
  authors: (
    (
      name: "David Sankel",
      institution: "Adobe",
      mail: "dsankel@adobe.com",
    ),
    (
      name: "Jon Bauman",
      institution: "Rust Foundation",
      mail: "jonbauman@rustfoundation.org",
    ),
  ),
  date: (
    year: 2025,
    month: "September",
    day: 30,
  ),
  abstract: [
      "Trivial Relocatability For C++ 26"@TrivialRelocatability introduced
      mechanisms for the identification and tagging of types whose objects can
      be "trivially" relocated from one memory address to another. It also
      included some standard library functions that perform this relocation.
      Useful as they are, these standard library functions are insufficient for
      important use cases such as `realloc` support, value representation
      serialization, and cross language interoperability. We propose completing
      the trivial relocation function set with the addition of a single function
      template, `std::restart_lifetime`, that addresses these unsupported use cases.

      #table(
        columns: (1fr,1fr),
        stroke: none,
        table.header[*Before*][*After*],
        table.hline(),
        [
        ```Cpp
        // A trivially relocatable, but not trivially
        // copyable, type.
        class Foo { /*...*/ };

        // Create a foo sequence with a single element using
        // Microsoft's specialized mimalloc allocator.
        void* foo_sequence_buffer =
          mi_malloc_aligned(sizeof(Foo), alignof(Foo));
        Foo* foo_sequence = new (foo_sequence_buffer) Foo();

        // Extend the sequence reusing the same memory if
        // possible
        foo_sequence_buffer = mi_realloc_aligned(
          foo_sequence, sizeof(Foo)*2, alignof(Foo));
        new (foo_sequence_buffer+sizeof(Foo)) Foo();
        foo_sequence = (Foo*)foo_sequence_buffer;




        foo_sequence[0].bar(); // Undefined behavior
        ```
        ],
        [
        ```Cpp









        // ...same as before...







        // Restart lifetime of relocated elements
        std::restart_lifetime(foo_sequence[0]);

        foo_sequence[0].bar(); // Okay
        ```
        ],
      )
  ],
)

// TODO: Consider another before/after table demonstrating improved Rust interop
// ergonomics.

= Introduction

It is a common-but-unspecified property that for many types, an object can be
relocated with a `memcpy` of its underlying bytes. Although the standard
guarantees this only for the small number of _trivially copyable_ types,
virtually all C++ compilers support `memcpy`-relocation of non-self-referential
types. Many applications have taken advantage of this property for performance
optimizations and a number of libraries have emerged that attempt to surface
this functionality in a generic way.
#footnote[See @TrivialRelocatability and @TrivialRelocatabilityAlt for a survey of such libraries.]

After much debate, a form of this functionality landed in the working
draft@CppStandardDraft with "Trivial Relocatability For C++
26"@TrivialRelocatability#footnote[See @TrivialRelocatabilityAlt for a notable
design alternative that was considered, but ultimately rejected.]. An important
trade-off in this design is that qualifying types may be "trivially" relocated
using only the `trivially_relocate` function; `memcpy` will not suffice.

```Cpp
// Foo is a trivially relocatable, but not
// trivially copyable, type.
class Foo { /*...*/ };
static_assert(
     std::is_trivially_relocatable_v<Foo>()
 && !std::is_trivially_copyable_v<Foo>());

void f() {
  alignas(Foo) char x1_buffer[sizeof(Foo)],
                    x2_buffer[sizeof(Foo)],
                    y1_buffer[sizeof(Foo)],
                    y2_buffer[sizeof(Foo)];

  // Relocating using std::memcpy results in
  // undefined behavior.
  Foo* x1 = new (x1_buffer) Foo();
  std::memcpy(&y1_buffer, x1, sizeof(Foo));
  Foo* y1 = reinterpret_cast<Foo*>(y1_buffer);
  y1->bar(); // Undefined behavior

  // Relocating using std::trivially_relocate
  // works as expected.
  Foo* x2 = new (x2_buffer) Foo();
  Foo* y2 = std::trivially_relocate(
    x2,
    x2+1,
    reinterpret_cast<Foo*>(&y2_buffer));
  y2->bar(); // Okay
}
```

One of the motivations for this trade-off is the ARM64e ABI which encodes an
object's address in its virtual table (vtable) pointer making
`memcpy`-relocation impossible for polymorphic types on this
platform#footnote[This is a memory safety vulnerability mitigation. See
@PointerAuthentication and @Arm64e for details.]. The requirement to call
`std::trivially_relocate` provides an opportunity for the standard library to
perform "fix ups" on these vtable pointers.

While `std::trivially_relocate` suffices for many use cases and neatly handles
the ARM64e platform, other important use-cases remain unaddressed. We propose to
compliment `std::trivially_relocate` with another function,
`std::restart_lifetime`, that addresses these use-cases and nicely compliments
`std::trivially_relocate`.

= Key `std::trivially_relocate` limitations <sec-usecases>

== `realloc` use case

Allocation libraries often feature a reallocation function (such as C's `realloc`)
that attempts to resize a given memory block#footnote[See mimalloc, jemalloc,
umm_malloc, and tcmalloc for some examples.
// TODO: Add references to these
]. It either extends the block
in-place or moves its contents to a new, larger allocation, freeing the original
block in the process.

Reallocation serves as an important performance optimization for
high-performance, low-level code that dynamically resizes arrays. By taking
advantage of the allocation library's knowledge of available space after the
originally allocated block, expensive copy operations and fragmentation can be
avoided.

However, because these reallocation functions potentially `memcpy`-relocate
objects, they may only portably be used with _trivially copyable_ types and
`std::trivially_relocate` will not help.

== Serialization use case

In-memory databases@InMemoryDatabase and tiered caching systems frequently
relocate data structures from memory to disk and back again. Unfortunately, this
operation is only possible for _trivially copyable_ types due to the lack of
sufficient library primitives for _trivially relocatable_ types.

== Specialized `memcpy` use case

A tuned memory copy operation can produce a 10% speedup over
`std::memcpy` and hetrogenious memory systems require an
alternative#footnote[See "Going faster than memcpy"@FastMemCpy and CUDA's
`cudaMemcpy`@CudaMemCpy for some notable examples.]. `std::trivially_relocate`'s
coupling of the physical moving of an object with restarting its lifetime makes
it is impossible to portably take advantage of these mechanisms with trivially
reloctable types.

== Rust-interop use case

// TODO: Jon: fill in

= `restart_lifetime`

We propose a `restart_lifetime` function that fits within the
`start_lifetime_as` series of functions. It allows us to separate the "memory
copying" aspect of relocation from restarting the object's lifetime at the new
memory address.

Here is an implementation of `std::trivially_relocate` using `restart_lifetime`
as a lower-level primitive.

```Cpp
template<class T>
requires /* ... */
T* trivially_relocate(T* first, T* last, T* result)
{
  std::memcpy( result,
               first,
               (last-first)*sizeof(T));
  for(size_t i = 0; i < (last-first); ++i)
    std::restart_lifetime(result[i]);
}
```

This separation of concerns enables developers to copy an object's value representation
to a new location by any means and then use it from the new location after a
call to `std::restart_lifetime`. This enables all the usecases highlighted in
@sec-usecases.

Here is an example of using `std::restart_lifetime` to roundtrip a `Foo` object
from main memory to GPU memory.

```Cpp
void * host_buffer = /*...*/
void * device_buffer = /*...*/

// Create a `Foo` object in host memory
Foo* x = new (host_buffer)[sizeof(Foo)];

// Move it to CUDA memory
cudaMemcpy( device_buffer,
            host_buffer,
            sizeof(Foo),
            cudaMemcpyHostToDevice );

// ... reuse host_buffer for other purposes

// Move it back to host memory
cudaMemcpy( host_buffer,
            device_buffer,
            sizeof(Foo),
            cudaMemcpyDevicetoHost );

// Restart the object's lifetime on the host
x = std::restart_lifetime<Foo>(host_buffer);

// ... continue using *x
```

// TODO: Create some examples that explain how start_lifetime_as would be used and demonstrate how it fixes things.

= Other considerations

// TODO: Fill this section. Review email thread to identify other discussions on this

== Will this cause security problems?

== Why is this being brought up now?

== Is this a bug fix or a feature?

== Is this critical for C++26?

== Should this _replace_ `trivially_relocate` instead of compliment it?

= Alternatives considered

// TODO: Add the alternative of ripping out trivial relocatability from the standard.

== `start_lifetime_at` extension

// TODO: Add prose explaining this and why it was discarded. We can probably get away
// with showing use cases instead of standardese.

#wg21.standardese[
```
template<class T>
T* start_lifetime_at(uintptr_t origin,
                     void* p) noexcept;
```

_Mandates_: `is_trivially_relocatable_v<T> && !is_const_v<T>` is `true`.

_Preconditions_:
- [`p`, `(char*)p + sizeof(T)`) denotes a region of allocated storage that
  is a subset of the region of storage reachable through [basic.compound] `p`
  and suitably aligned for the type `T`.
- The contents of [`p`, `(char*)p + sizeof(T)`) is the value representation of
  an object `a` that was stored at `origin`.

_Effects_: Implicitly creates an object _b_ within the denoted region of type
`T` whose address is `p`, whose lifetime has begun, and whose object
representation is the same as that of _a_.

_Returns_: A pointer to the _b_ defined in the _Effects_ paragraph.
]

= Wording

#wg21.standardese[
```
template<class T>
T* restart_lifetime(void* p) noexcept;
```

_Mandates_: `is_trivially_relocatable_v<T> && !is_const_v<T>` is `true`.

_Preconditions_:
- [`p`, `(char*)p + sizeof(T)`) denotes a region of allocated storage that
  is a subset of the region of storage reachable through [basic.compound] `p`
  and suitably aligned for the type `T`.
- The contents of [`p`, `(char*)p + sizeof(T)`) is the value representation of
  an object _a_ that was stored at another address.

_Effects_: Implicitly creates an object _b_ within the denoted region of type
`T` whose address is `p`, whose lifetime has begun, and whose object
representation is the same as that of _a_. If _a_ was still within its lifetime,
its lifetime is ended.

_Returns_: A pointer to the _b_ defined in the _Effects_ paragraph.
]

= Conclusion

// TODO: Add conclusion

= Acknowledgments

// TODO: Mention Pablo Halpern and others who contributed to the conversation.

#bibliography("references.yml", style: "ieee")
