// Talos-XII Binary Codec: a zero-dependency serde-based binary serializer/deserializer.
// Replaces bincode with a minimal, vulnerability-free implementation.
//
// Wire format (little-endian throughout):
//   bool        -> 1 byte (0x00 / 0x01)
//   i8/u8       -> 1 byte
//   i16/u16     -> 2 bytes
//   i32/u32     -> 4 bytes
//   i64/u64     -> 8 bytes
//   f32         -> 4 bytes (IEEE 754)
//   f64         -> 8 bytes (IEEE 754)
//   str/bytes   -> u64 length prefix + raw bytes
//   seq/tuple   -> u64 length prefix + elements
//   map         -> u64 length prefix + key-value pairs
//   struct      -> fields in order (no field names)
//   enum        -> u32 variant index + payload
//   option      -> 0x00 (None) | 0x01 + value (Some)

use serde::{de, ser, Deserialize, Serialize};
use std::fmt;
use std::io::{self, Read, Write};

/// Hard cap on deserialized collection length to prevent OOM from corrupt data.
/// 32M elements is generous for any realistic model payload.
const MAX_COLLECTION_LEN: usize = 32 * 1024 * 1024;

// ═══════════════════════════════════════════════════════════════════════════
//  Error
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Message(String),
    UnexpectedEof,
    InvalidBool(u8),
    InvalidUtf8,
    InvalidEnumVariant(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::Message(s) => write!(f, "{}", s),
            Error::UnexpectedEof => write!(f, "unexpected end of input"),
            Error::InvalidBool(v) => write!(f, "invalid bool byte: {}", v),
            Error::InvalidUtf8 => write!(f, "invalid UTF-8 in string"),
            Error::InvalidEnumVariant(v) => write!(f, "invalid enum variant: {}", v),
        }
    }
}

impl std::error::Error for Error {}

impl ser::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error::Message(msg.to_string())
    }
}

impl de::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error::Message(msg.to_string())
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════════

pub fn serialize_into<W: Write, T: Serialize>(writer: W, value: &T) -> Result<(), Error> {
    let mut ser = BinSerializer { writer };
    value.serialize(&mut ser)
}

pub fn deserialize_from<R: Read, T: for<'de> Deserialize<'de>>(reader: R) -> Result<T, Error> {
    let mut de = BinDeserializer { reader };
    T::deserialize(&mut de)
}

#[allow(dead_code)]
pub fn to_vec<T: Serialize>(value: &T) -> Result<Vec<u8>, Error> {
    let mut buf = Vec::new();
    serialize_into(&mut buf, value)?;
    Ok(buf)
}

#[allow(dead_code)]
pub fn from_slice<T: for<'de> Deserialize<'de>>(data: &[u8]) -> Result<T, Error> {
    deserialize_from(data)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Serializer
// ═══════════════════════════════════════════════════════════════════════════

struct BinSerializer<W: Write> {
    writer: W,
}

impl<W: Write> BinSerializer<W> {
    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> Result<(), Error> {
        self.writer.write_all(buf).map_err(Error::Io)
    }
}

impl<W: Write> ser::Serializer for &mut BinSerializer<W> {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fn serialize_bool(self, v: bool) -> Result<(), Error> {
        self.write_all(&[v as u8])
    }

    fn serialize_i8(self, v: i8) -> Result<(), Error> {
        self.write_all(&v.to_le_bytes())
    }

    fn serialize_i16(self, v: i16) -> Result<(), Error> {
        self.write_all(&v.to_le_bytes())
    }

    fn serialize_i32(self, v: i32) -> Result<(), Error> {
        self.write_all(&v.to_le_bytes())
    }

    fn serialize_i64(self, v: i64) -> Result<(), Error> {
        self.write_all(&v.to_le_bytes())
    }

    fn serialize_u8(self, v: u8) -> Result<(), Error> {
        self.write_all(&[v])
    }

    fn serialize_u16(self, v: u16) -> Result<(), Error> {
        self.write_all(&v.to_le_bytes())
    }

    fn serialize_u32(self, v: u32) -> Result<(), Error> {
        self.write_all(&v.to_le_bytes())
    }

    fn serialize_u64(self, v: u64) -> Result<(), Error> {
        self.write_all(&v.to_le_bytes())
    }

    fn serialize_f32(self, v: f32) -> Result<(), Error> {
        debug_assert!(v.is_finite(), "serializing non-finite f32: {}", v);
        self.write_all(&v.to_le_bytes())
    }

    fn serialize_f64(self, v: f64) -> Result<(), Error> {
        debug_assert!(v.is_finite(), "serializing non-finite f64: {}", v);
        self.write_all(&v.to_le_bytes())
    }

    fn serialize_char(self, v: char) -> Result<(), Error> {
        self.serialize_u32(v as u32)
    }

    fn serialize_str(self, v: &str) -> Result<(), Error> {
        self.write_all(&(v.len() as u64).to_le_bytes())?;
        self.write_all(v.as_bytes())
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<(), Error> {
        self.write_all(&(v.len() as u64).to_le_bytes())?;
        self.write_all(v)
    }

    fn serialize_none(self) -> Result<(), Error> {
        self.write_all(&[0u8])
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), Error> {
        self.write_all(&[1u8])?;
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), Error> {
        Ok(())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<(), Error> {
        Ok(())
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
    ) -> Result<(), Error> {
        self.write_all(&variant_index.to_le_bytes())
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.write_all(&variant_index.to_le_bytes())?;
        value.serialize(self)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Error> {
        let len = len.ok_or_else(|| Error::Message("sequence length required".into()))?;
        self.write_all(&(len as u64).to_le_bytes())?;
        Ok(self)
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Error> {
        Ok(self)
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Error> {
        Ok(self)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Error> {
        self.write_all(&variant_index.to_le_bytes())?;
        Ok(self)
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Error> {
        let len = len.ok_or_else(|| Error::Message("map length required".into()))?;
        self.write_all(&(len as u64).to_le_bytes())?;
        Ok(self)
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Error> {
        Ok(self)
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Error> {
        self.write_all(&variant_index.to_le_bytes())?;
        Ok(self)
    }
}

// --- Compound serialization trait impls ---

impl<W: Write> ser::SerializeSeq for &mut BinSerializer<W> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

impl<W: Write> ser::SerializeTuple for &mut BinSerializer<W> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

impl<W: Write> ser::SerializeTupleStruct for &mut BinSerializer<W> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

impl<W: Write> ser::SerializeTupleVariant for &mut BinSerializer<W> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

impl<W: Write> ser::SerializeMap for &mut BinSerializer<W> {
    type Ok = ();
    type Error = Error;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Error> {
        key.serialize(&mut **self)
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

impl<W: Write> ser::SerializeStruct for &mut BinSerializer<W> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

impl<W: Write> ser::SerializeStructVariant for &mut BinSerializer<W> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Deserializer
// ═══════════════════════════════════════════════════════════════════════════

struct BinDeserializer<R: Read> {
    reader: R,
}

impl<R: Read> BinDeserializer<R> {
    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), Error> {
        self.reader.read_exact(buf).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                Error::UnexpectedEof
            } else {
                Error::Io(e)
            }
        })
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8, Error> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    #[inline]
    fn read_u16(&mut self) -> Result<u16, Error> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    #[inline]
    fn read_u32(&mut self) -> Result<u32, Error> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    #[inline]
    fn read_u64(&mut self) -> Result<u64, Error> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    #[inline]
    fn read_i8(&mut self) -> Result<i8, Error> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(i8::from_le_bytes(buf))
    }

    #[inline]
    fn read_i16(&mut self) -> Result<i16, Error> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    #[inline]
    fn read_i32(&mut self) -> Result<i32, Error> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    #[inline]
    fn read_i64(&mut self) -> Result<i64, Error> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    #[inline]
    fn read_f32(&mut self) -> Result<f32, Error> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    #[inline]
    fn read_f64(&mut self) -> Result<f64, Error> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, Error> {
        let len = self.read_u64()? as usize;
        // Sanity cap to prevent OOM on corrupt data
        if len > 256 * 1024 * 1024 {
            return Err(Error::Message(format!("sequence too large: {} bytes", len)));
        }
        let mut buf = vec![0u8; len];
        self.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn read_string(&mut self) -> Result<String, Error> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes).map_err(|_| Error::InvalidUtf8)
    }
}

impl<'de, R: Read> de::Deserializer<'de> for &mut BinDeserializer<R> {
    type Error = Error;

    fn deserialize_any<V: de::Visitor<'de>>(self, _visitor: V) -> Result<V::Value, Error> {
        Err(Error::Message(
            "binary_codec does not support deserialize_any".into(),
        ))
    }

    fn deserialize_bool<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let b = self.read_u8()?;
        match b {
            0 => visitor.visit_bool(false),
            1 => visitor.visit_bool(true),
            v => Err(Error::InvalidBool(v)),
        }
    }

    fn deserialize_i8<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_i8(self.read_i8()?)
    }

    fn deserialize_i16<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_i16(self.read_i16()?)
    }

    fn deserialize_i32<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_i32(self.read_i32()?)
    }

    fn deserialize_i64<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_i64(self.read_i64()?)
    }

    fn deserialize_u8<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_u8(self.read_u8()?)
    }

    fn deserialize_u16<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_u16(self.read_u16()?)
    }

    fn deserialize_u32<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_u32(self.read_u32()?)
    }

    fn deserialize_u64<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_u64(self.read_u64()?)
    }

    fn deserialize_f32<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_f32(self.read_f32()?)
    }

    fn deserialize_f64<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_f64(self.read_f64()?)
    }

    fn deserialize_char<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let code = self.read_u32()?;
        let c = char::from_u32(code)
            .ok_or_else(|| Error::Message(format!("invalid char code: {}", code)))?;
        visitor.visit_char(c)
    }

    fn deserialize_str<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let s = self.read_string()?;
        visitor.visit_string(s)
    }

    fn deserialize_string<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let s = self.read_string()?;
        visitor.visit_string(s)
    }

    fn deserialize_bytes<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let bytes = self.read_bytes()?;
        visitor.visit_byte_buf(bytes)
    }

    fn deserialize_byte_buf<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let bytes = self.read_bytes()?;
        visitor.visit_byte_buf(bytes)
    }

    fn deserialize_option<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let tag = self.read_u8()?;
        match tag {
            0 => visitor.visit_none(),
            1 => visitor.visit_some(self),
            v => Err(Error::InvalidBool(v)),
        }
    }

    fn deserialize_unit<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_unit()
    }

    fn deserialize_unit_struct<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_unit()
    }

    fn deserialize_newtype_struct<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let len = self.read_u64()? as usize;
        if len > MAX_COLLECTION_LEN {
            return Err(Error::Message(format!(
                "sequence length {} exceeds cap {}",
                len, MAX_COLLECTION_LEN
            )));
        }
        visitor.visit_seq(SeqAccess {
            de: self,
            remaining: len,
        })
    }

    fn deserialize_tuple<V: de::Visitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_seq(SeqAccess {
            de: self,
            remaining: len,
        })
    }

    fn deserialize_tuple_struct<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_seq(SeqAccess {
            de: self,
            remaining: len,
        })
    }

    fn deserialize_map<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let len = self.read_u64()? as usize;
        if len > MAX_COLLECTION_LEN {
            return Err(Error::Message(format!(
                "map length {} exceeds cap {}",
                len, MAX_COLLECTION_LEN
            )));
        }
        visitor.visit_map(MapAccess {
            de: self,
            remaining: len,
        })
    }

    fn deserialize_struct<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_seq(SeqAccess {
            de: self,
            remaining: fields.len(),
        })
    }

    fn deserialize_enum<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_enum(EnumAccess {
            de: self,
            num_variants: variants.len(),
        })
    }

    fn deserialize_identifier<V: de::Visitor<'de>>(self, _visitor: V) -> Result<V::Value, Error> {
        Err(Error::Message(
            "binary_codec does not support deserialize_identifier".into(),
        ))
    }

    fn deserialize_ignored_any<V: de::Visitor<'de>>(self, _visitor: V) -> Result<V::Value, Error> {
        Err(Error::Message(
            "binary_codec does not support deserialize_ignored_any".into(),
        ))
    }
}

// --- SeqAccess ---

struct SeqAccess<'a, R: Read> {
    de: &'a mut BinDeserializer<R>,
    remaining: usize,
}

impl<'de, 'a, R: Read> de::SeqAccess<'de> for SeqAccess<'a, R> {
    type Error = Error;

    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Error> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(&mut *self.de).map(Some)
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining)
    }
}

// --- MapAccess ---

struct MapAccess<'a, R: Read> {
    de: &'a mut BinDeserializer<R>,
    remaining: usize,
}

impl<'de, 'a, R: Read> de::MapAccess<'de> for MapAccess<'a, R> {
    type Error = Error;

    fn next_key_seed<K: de::DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, Error> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(&mut *self.de).map(Some)
    }

    fn next_value_seed<V: de::DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, Error> {
        seed.deserialize(&mut *self.de)
    }
}

// --- EnumAccess ---

struct EnumAccess<'a, R: Read> {
    de: &'a mut BinDeserializer<R>,
    num_variants: usize,
}

impl<'de, 'a, R: Read> de::EnumAccess<'de> for EnumAccess<'a, R> {
    type Error = Error;
    type Variant = VariantAccess<'a, R>;

    fn variant_seed<V: de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant), Error> {
        let variant_index = self.de.read_u32()?;
        if self.num_variants > 0 && variant_index as usize >= self.num_variants {
            return Err(Error::InvalidEnumVariant(variant_index));
        }
        let val = seed.deserialize(VariantIndexDeserializer {
            index: variant_index,
        })?;
        Ok((val, VariantAccess { de: self.de }))
    }
}

struct VariantAccess<'a, R: Read> {
    de: &'a mut BinDeserializer<R>,
}

impl<'de, 'a, R: Read> de::VariantAccess<'de> for VariantAccess<'a, R> {
    type Error = Error;

    fn unit_variant(self) -> Result<(), Error> {
        Ok(())
    }

    fn newtype_variant_seed<T: de::DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, Error> {
        seed.deserialize(self.de)
    }

    fn tuple_variant<V: de::Visitor<'de>>(self, len: usize, visitor: V) -> Result<V::Value, Error> {
        de::Deserializer::deserialize_tuple(self.de, len, visitor)
    }

    fn struct_variant<V: de::Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        de::Deserializer::deserialize_struct(self.de, "", fields, visitor)
    }
}

// Tiny helper deserializer that yields a u32 variant index
struct VariantIndexDeserializer {
    index: u32,
}

impl<'de> de::Deserializer<'de> for VariantIndexDeserializer {
    type Error = Error;

    fn deserialize_any<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_u32(self.index)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct
        map struct enum identifier ignored_any
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[test]
    fn roundtrip_primitives() {
        let v: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let buf = to_vec(&v).unwrap();
        assert_eq!(buf.len(), 8);
        let decoded: u64 = from_slice(&buf).unwrap();
        assert_eq!(v, decoded);

        let f: f64 = std::f64::consts::PI;
        let buf = to_vec(&f).unwrap();
        let decoded: f64 = from_slice(&buf).unwrap();
        assert_eq!(f, decoded);

        let b: bool = true;
        let buf = to_vec(&b).unwrap();
        assert_eq!(buf, vec![1]);
        let decoded: bool = from_slice(&buf).unwrap();
        assert_eq!(b, decoded);
    }

    #[test]
    fn roundtrip_string() {
        let s = "Hello, 终末地!".to_string();
        let buf = to_vec(&s).unwrap();
        let decoded: String = from_slice(&buf).unwrap();
        assert_eq!(s, decoded);
    }

    #[test]
    fn roundtrip_vec() {
        let v: Vec<f64> = vec![1.0, 2.0, std::f64::consts::PI, -42.0, 0.0];
        let buf = to_vec(&v).unwrap();
        let decoded: Vec<f64> = from_slice(&buf).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn roundtrip_nested_struct() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Inner {
            weights: Vec<f64>,
            name: String,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Outer {
            inner: Inner,
            bias: Option<Vec<f64>>,
            count: u32,
            flag: bool,
        }

        let val = Outer {
            inner: Inner {
                weights: vec![0.1, 0.2, 0.3],
                name: "layer_0".into(),
            },
            bias: Some(vec![0.0, 0.0, 0.0]),
            count: 42,
            flag: true,
        };

        let buf = to_vec(&val).unwrap();
        let decoded: Outer = from_slice(&buf).unwrap();
        assert_eq!(val, decoded);
    }

    #[test]
    fn roundtrip_option_none() {
        let val: Option<Vec<f64>> = None;
        let buf = to_vec(&val).unwrap();
        let decoded: Option<Vec<f64>> = from_slice(&buf).unwrap();
        assert_eq!(val, decoded);
    }

    #[test]
    fn roundtrip_enum() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        enum Mode {
            Fast,
            Normal,
            Custom(u32),
            Named { x: f64, y: f64 },
        }

        for val in [
            Mode::Fast,
            Mode::Normal,
            Mode::Custom(7),
            Mode::Named { x: 1.0, y: 2.0 },
        ] {
            let buf = to_vec(&val).unwrap();
            let decoded: Mode = from_slice(&buf).unwrap();
            assert_eq!(val, decoded);
        }
    }

    #[test]
    fn roundtrip_large_vec_f64() {
        let v: Vec<f64> = (0..10000).map(|i| i as f64 * 0.001).collect();
        let buf = to_vec(&v).unwrap();
        // 8 bytes length prefix + 10000 * 8 bytes
        assert_eq!(buf.len(), 8 + 10000 * 8);
        let decoded: Vec<f64> = from_slice(&buf).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn roundtrip_streaming_io() {
        let original: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut buf = Vec::new();
        serialize_into(&mut buf, &original).unwrap();
        let decoded: Vec<f64> = deserialize_from(&buf[..]).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn error_on_truncated_data() {
        let buf = vec![0u8; 3]; // too short for a u64
        let result: Result<u64, _> = from_slice(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn roundtrip_map() {
        use std::collections::HashMap;
        let mut m = HashMap::new();
        m.insert("alpha".to_string(), 1.0_f64);
        m.insert("beta".to_string(), 2.0);
        let buf = to_vec(&m).unwrap();
        let decoded: HashMap<String, f64> = from_slice(&buf).unwrap();
        assert_eq!(m, decoded);
    }
}
